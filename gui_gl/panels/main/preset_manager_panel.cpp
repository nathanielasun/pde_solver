#include "preset_manager_panel.h"
#include "imgui.h"
#include "templates.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace {
using json = nlohmann::json;

// Persistent UI state across frames
static bool showSaveInput = false;
static char presetNameBuf[256] = "";
static std::string saveStatusMsg;
static float saveStatusTimer = 0.0f;

const char* BcKindLabel(int kind) {
  switch (kind) {
    case 0: return "dirichlet";
    case 1: return "neumann";
    case 2: return "robin";
    default: return "dirichlet";
  }
}

int BcKindFromString(const std::string& s) {
  if (s == "neumann") return 1;
  if (s == "robin") return 2;
  return 0;  // default dirichlet
}

json BoundaryToJson(const BoundaryInput& bc) {
  json j;
  j["kind"] = BcKindLabel(bc.kind);
  j["value"] = bc.value;
  if (bc.kind == 2) {  // Robin has extra params
    j["alpha"] = bc.alpha;
    j["beta"] = bc.beta;
    j["gamma"] = bc.gamma;
  }
  return j;
}

void BoundaryFromJson(const json& j, BoundaryInput& bc) {
  if (j.contains("kind") && j["kind"].is_string()) {
    bc.kind = BcKindFromString(j["kind"].get<std::string>());
  }
  if (j.contains("value") && j["value"].is_string()) {
    bc.value = j["value"].get<std::string>();
  }
  if (j.contains("alpha") && j["alpha"].is_string()) {
    bc.alpha = j["alpha"].get<std::string>();
  }
  if (j.contains("beta") && j["beta"].is_string()) {
    bc.beta = j["beta"].get<std::string>();
  }
  if (j.contains("gamma") && j["gamma"].is_string()) {
    bc.gamma = j["gamma"].get<std::string>();
  }
}

json SerializePreset(const PresetManagerPanelState& s) {
  json root;
  root["schema_version"] = 1;
  root["pde_text"] = s.pde_text;

  json domain;
  domain["xmin"] = s.bound_xmin;
  domain["xmax"] = s.bound_xmax;
  domain["ymin"] = s.bound_ymin;
  domain["ymax"] = s.bound_ymax;
  domain["zmin"] = s.bound_zmin;
  domain["zmax"] = s.bound_zmax;
  root["domain"] = domain;

  json grid;
  grid["nx"] = s.grid_nx;
  grid["ny"] = s.grid_ny;
  grid["nz"] = s.grid_nz;
  root["grid"] = grid;

  root["coord_mode"] = s.coord_mode;
  root["method_index"] = s.method_index;
  root["tolerance"] = s.solver_tol;
  root["max_iterations"] = s.solver_max_iter;

  json boundary;
  boundary["left"] = BoundaryToJson(s.bc_left);
  boundary["right"] = BoundaryToJson(s.bc_right);
  boundary["bottom"] = BoundaryToJson(s.bc_bottom);
  boundary["top"] = BoundaryToJson(s.bc_top);
  boundary["front"] = BoundaryToJson(s.bc_front);
  boundary["back"] = BoundaryToJson(s.bc_back);
  root["boundary"] = boundary;

  return root;
}

bool DeserializePreset(const json& root, PresetManagerPanelState& s) {
  if (root.contains("pde_text") && root["pde_text"].is_string()) {
    s.pde_text = root["pde_text"].get<std::string>();
  }

  if (root.contains("domain") && root["domain"].is_object()) {
    const auto& d = root["domain"];
    if (d.contains("xmin") && d["xmin"].is_number()) s.bound_xmin = d["xmin"].get<double>();
    if (d.contains("xmax") && d["xmax"].is_number()) s.bound_xmax = d["xmax"].get<double>();
    if (d.contains("ymin") && d["ymin"].is_number()) s.bound_ymin = d["ymin"].get<double>();
    if (d.contains("ymax") && d["ymax"].is_number()) s.bound_ymax = d["ymax"].get<double>();
    if (d.contains("zmin") && d["zmin"].is_number()) s.bound_zmin = d["zmin"].get<double>();
    if (d.contains("zmax") && d["zmax"].is_number()) s.bound_zmax = d["zmax"].get<double>();
  }

  if (root.contains("grid") && root["grid"].is_object()) {
    const auto& g = root["grid"];
    if (g.contains("nx") && g["nx"].is_number_integer()) s.grid_nx = g["nx"].get<int>();
    if (g.contains("ny") && g["ny"].is_number_integer()) s.grid_ny = g["ny"].get<int>();
    if (g.contains("nz") && g["nz"].is_number_integer()) s.grid_nz = g["nz"].get<int>();
  }

  if (root.contains("coord_mode") && root["coord_mode"].is_number_integer()) {
    s.coord_mode = root["coord_mode"].get<int>();
  }
  if (root.contains("method_index") && root["method_index"].is_number_integer()) {
    s.method_index = root["method_index"].get<int>();
  }
  if (root.contains("tolerance") && root["tolerance"].is_number()) {
    s.solver_tol = root["tolerance"].get<double>();
  }
  if (root.contains("max_iterations") && root["max_iterations"].is_number_integer()) {
    s.solver_max_iter = root["max_iterations"].get<int>();
  }

  if (root.contains("boundary") && root["boundary"].is_object()) {
    const auto& b = root["boundary"];
    if (b.contains("left")) BoundaryFromJson(b["left"], s.bc_left);
    if (b.contains("right")) BoundaryFromJson(b["right"], s.bc_right);
    if (b.contains("bottom")) BoundaryFromJson(b["bottom"], s.bc_bottom);
    if (b.contains("top")) BoundaryFromJson(b["top"], s.bc_top);
    if (b.contains("front")) BoundaryFromJson(b["front"], s.bc_front);
    if (b.contains("back")) BoundaryFromJson(b["back"], s.bc_back);
  }

  return true;
}

bool SavePresetFile(const std::filesystem::path& path, const json& data,
                    std::string* error) {
  std::error_code ec;
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
      if (error) *error = "failed to create directory: " + ec.message();
      return false;
    }
  }
  std::ofstream file(path, std::ios::trunc);
  if (!file) {
    if (error) *error = "failed to open file for writing: " + path.string();
    return false;
  }
  file << data.dump(2);
  if (!file.good()) {
    if (error) *error = "failed to write preset file: " + path.string();
    return false;
  }
  return true;
}

bool LoadPresetFile(const std::filesystem::path& path, json* data,
                    std::string* error) {
  std::ifstream file(path);
  if (!file) {
    if (error) *error = "failed to open file: " + path.string();
    return false;
  }
  try {
    *data = json::parse(file);
  } catch (const json::parse_error& e) {
    if (error) *error = std::string("JSON parse error: ") + e.what();
    return false;
  }
  return true;
}

struct UserPresetEntry {
  std::string name;
  std::filesystem::path path;
};

std::vector<UserPresetEntry> ScanUserPresets(const std::string& directory) {
  std::vector<UserPresetEntry> entries;
  std::error_code ec;
  if (!std::filesystem::exists(directory, ec) ||
      !std::filesystem::is_directory(directory, ec)) {
    return entries;
  }
  for (const auto& entry : std::filesystem::directory_iterator(directory, ec)) {
    if (entry.is_regular_file() && entry.path().extension() == ".json") {
      UserPresetEntry e;
      e.name = entry.path().stem().string();
      e.path = entry.path();
      entries.push_back(e);
    }
  }
  std::sort(entries.begin(), entries.end(),
            [](const UserPresetEntry& a, const UserPresetEntry& b) {
              return a.name < b.name;
            });
  return entries;
}

// Sanitize a file name: strip characters that are unsafe for file paths
std::string SanitizeFileName(const std::string& input) {
  std::string result;
  result.reserve(input.size());
  for (char c : input) {
    if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' ||
        c == '"' || c == '<' || c == '>' || c == '|') {
      result += '_';
    } else {
      result += c;
    }
  }
  // Trim leading/trailing whitespace
  size_t start = result.find_first_not_of(" \t");
  size_t end = result.find_last_not_of(" \t");
  if (start == std::string::npos) return "";
  return result.substr(start, end - start + 1);
}

}  // namespace

void RenderPresetManagerPanel(PresetManagerPanelState& state,
                               const std::vector<std::string>& components) {
  ImGui::Text("Preset Manager");
  ImGui::Separator();
  ImGui::Spacing();

  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                     "Save and load complete problem setups.");
  ImGui::Spacing();

  // --- Built-in Presets ---
  if (ImGui::CollapsingHeader("Built-in Presets", ImGuiTreeNodeFlags_DefaultOpen)) {
    auto templates = GetProblemTemplates();
    for (const auto& t : templates) {
      ImGui::PushID(t.name.c_str());
      bool selected = false;
      if (ImGui::Selectable(t.name.c_str(), &selected, ImGuiSelectableFlags_None)) {
        // Apply built-in template to state
        state.pde_text = t.pde_latex;

        // Parse domain bounds
        std::istringstream boundsStream(t.domain_bounds);
        std::string token;
        std::vector<double> bounds;
        while (std::getline(boundsStream, token, ',')) {
          bounds.push_back(std::stod(token));
        }
        if (bounds.size() >= 4) {
          state.bound_xmin = bounds[0];
          state.bound_xmax = bounds[1];
          state.bound_ymin = bounds[2];
          state.bound_ymax = bounds[3];
          if (bounds.size() >= 6) {
            state.bound_zmin = bounds[4];
            state.bound_zmax = bounds[5];
          }
        }

        // Parse grid resolution
        std::istringstream gridStream(t.grid_resolution);
        std::vector<int> gridVals;
        while (std::getline(gridStream, token, ',')) {
          gridVals.push_back(std::stoi(token));
        }
        if (gridVals.size() >= 2) {
          state.grid_nx = gridVals[0];
          state.grid_ny = gridVals[1];
          if (gridVals.size() >= 3) {
            state.grid_nz = gridVals[2];
          }
        }

        // Coordinate mode
        switch (t.coord_system) {
          case CoordinateSystem::Cartesian: state.coord_mode = 0; break;
          case CoordinateSystem::Polar: state.coord_mode = 2; break;
          case CoordinateSystem::Axisymmetric: state.coord_mode = 3; break;
          case CoordinateSystem::Cylindrical: state.coord_mode = 4; break;
          case CoordinateSystem::SphericalSurface: state.coord_mode = 5; break;
          case CoordinateSystem::SphericalVolume: state.coord_mode = 6; break;
          case CoordinateSystem::ToroidalSurface: state.coord_mode = 7; break;
          case CoordinateSystem::ToroidalVolume: state.coord_mode = 8; break;
          default: state.coord_mode = 0; break;
        }

        // Solver method
        switch (t.recommended_method) {
          case SolveMethod::Jacobi: state.method_index = 0; break;
          case SolveMethod::GaussSeidel: state.method_index = 1; break;
          case SolveMethod::SOR: state.method_index = 2; break;
          case SolveMethod::CG: state.method_index = 3; break;
          case SolveMethod::BiCGStab: state.method_index = 4; break;
          case SolveMethod::GMRES: state.method_index = 5; break;
          case SolveMethod::MultigridVcycle: state.method_index = 6; break;
          default: state.method_index = 0; break;
        }

        // Boundary conditions
        state.bc_left.kind = t.bc_left_kind;
        state.bc_left.value = t.bc_left;
        state.bc_right.kind = t.bc_right_kind;
        state.bc_right.value = t.bc_right;
        state.bc_bottom.kind = t.bc_bottom_kind;
        state.bc_bottom.value = t.bc_bottom;
        state.bc_top.kind = t.bc_top_kind;
        state.bc_top.value = t.bc_top;
        state.bc_front.kind = t.bc_front_kind;
        state.bc_front.value = t.bc_front;
        state.bc_back.kind = t.bc_back_kind;
        state.bc_back.value = t.bc_back;
      }
      if (!t.description.empty() && ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s", t.description.c_str());
      }
      ImGui::PopID();
    }
  }

  ImGui::Spacing();

  // --- User Presets ---
  if (ImGui::CollapsingHeader("User Presets", ImGuiTreeNodeFlags_DefaultOpen)) {
    auto userPresets = ScanUserPresets(state.preset_directory);
    if (userPresets.empty()) {
      ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No user presets saved.");
    } else {
      static int deleteConfirmIdx = -1;
      for (int i = 0; i < static_cast<int>(userPresets.size()); ++i) {
        const auto& entry = userPresets[static_cast<size_t>(i)];
        ImGui::PushID(i);

        // Preset name as a selectable label
        ImGui::AlignTextToFramePadding();
        ImGui::BulletText("%s", entry.name.c_str());

        // Load button
        ImGui::SameLine();
        if (ImGui::SmallButton("Load")) {
          json data;
          std::string loadError;
          if (LoadPresetFile(entry.path, &data, &loadError)) {
            if (DeserializePreset(data, state)) {
              saveStatusMsg = "Loaded: " + entry.name;
              saveStatusTimer = 3.0f;
            } else {
              saveStatusMsg = "Error applying preset";
              saveStatusTimer = 4.0f;
            }
          } else {
            saveStatusMsg = "Load error: " + loadError;
            saveStatusTimer = 4.0f;
          }
          deleteConfirmIdx = -1;
        }

        // Delete button
        ImGui::SameLine();
        if (deleteConfirmIdx == i) {
          ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Confirm?");
          ImGui::SameLine();
          if (ImGui::SmallButton("Yes")) {
            std::error_code ec;
            std::filesystem::remove(entry.path, ec);
            if (!ec) {
              saveStatusMsg = "Deleted: " + entry.name;
            } else {
              saveStatusMsg = "Delete failed: " + ec.message();
            }
            saveStatusTimer = 3.0f;
            deleteConfirmIdx = -1;
          }
          ImGui::SameLine();
          if (ImGui::SmallButton("No")) {
            deleteConfirmIdx = -1;
          }
        } else {
          if (ImGui::SmallButton("Delete")) {
            deleteConfirmIdx = i;
          }
        }

        ImGui::PopID();
      }
    }
  }

  ImGui::Spacing();
  ImGui::Separator();

  // --- Save Current Setup ---
  if (!showSaveInput) {
    if (ImGui::Button("Save Current Setup", ImVec2(-1, 0))) {
      showSaveInput = true;
      presetNameBuf[0] = '\0';
    }
  } else {
    ImGui::Text("Preset Name:");
    ImGui::SetNextItemWidth(-1);
    bool enterPressed = ImGui::InputText("##preset_name", presetNameBuf,
                                         sizeof(presetNameBuf),
                                         ImGuiInputTextFlags_EnterReturnsTrue);
    if (ImGui::Button("Save", ImVec2(80, 0)) || enterPressed) {
      std::string rawName(presetNameBuf);
      std::string safeName = SanitizeFileName(rawName);
      if (safeName.empty()) {
        saveStatusMsg = "Please enter a valid preset name.";
        saveStatusTimer = 3.0f;
      } else {
        std::filesystem::path presetPath =
            std::filesystem::path(state.preset_directory) / (safeName + ".json");
        json data = SerializePreset(state);
        std::string writeError;
        if (SavePresetFile(presetPath, data, &writeError)) {
          saveStatusMsg = "Saved: " + safeName;
          saveStatusTimer = 3.0f;
          showSaveInput = false;
          presetNameBuf[0] = '\0';
        } else {
          saveStatusMsg = "Save error: " + writeError;
          saveStatusTimer = 4.0f;
        }
      }
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(80, 0))) {
      showSaveInput = false;
      presetNameBuf[0] = '\0';
    }
  }

  // --- Status message ---
  if (saveStatusTimer > 0.0f) {
    saveStatusTimer -= ImGui::GetIO().DeltaTime;
    ImVec4 color = (saveStatusMsg.find("error") != std::string::npos ||
                    saveStatusMsg.find("Error") != std::string::npos ||
                    saveStatusMsg.find("failed") != std::string::npos ||
                    saveStatusMsg.find("Please") != std::string::npos)
        ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f)
        : ImVec4(0.3f, 0.9f, 0.4f, 1.0f);
    ImGui::TextColored(color, "%s", saveStatusMsg.c_str());
  }
}
