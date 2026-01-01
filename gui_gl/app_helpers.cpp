#include "app_helpers.h"
#include "app_state.h"
#include "vtk_io.h"
#include "boundary_utils.h"
#include "backend.h"
#include "pde_types.h"
#include "input_parse.h"
#include "utils/string_utils.h"
#include "utils/math_utils.h"
#include "utils/coordinate_utils.h"
#include "utils/path_utils.h"
#include "io/preferences_io.h"
#include "io/file_utils.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <OpenGL/gl3.h>
#include "stb_image.h"
#include "imgui.h"

extern char** environ;

// Constants
namespace {
constexpr int kLogLimit = 300;
}

// Expression remapping
std::string RemapSphericalSurfaceExpr(const std::string& input) {
  std::string out = input;
  ReplaceAll(&out, "\\vartheta", "x");
  ReplaceAll(&out, "\\theta", "x");
  ReplaceAll(&out, "vartheta", "x");
  ReplaceAll(&out, "theta", "x");
  ReplaceAll(&out, "\\varphi", "y");
  ReplaceAll(&out, "\\phi", "y");
  ReplaceAll(&out, "varphi", "y");
  ReplaceAll(&out, "phi", "y");
  return out;
}

std::string RemapAxisymmetricExpr(const std::string& input) {
  std::string out = input;
  ReplaceAll(&out, "\\partial z", "\\partial y");
  ReplaceAll(&out, "\\partial_z", "\\partial_y");
  ReplaceAll(&out, "\\frac{\\partial u}{\\partial z}", "\\frac{\\partial u}{\\partial y}");
  ReplaceAll(&out, "\\frac{\\partial^2 u}{\\partial z^2}", "\\frac{\\partial^2 u}{\\partial y^2}");
  ReplaceAll(&out, "\\frac{\\partial^2 u}{\\partial z \\partial z}",
             "\\frac{\\partial^2 u}{\\partial y \\partial y}");
  ReplaceAll(&out, "\\frac{\\partial^2 u}{\\partial z\\partial z}",
             "\\frac{\\partial^2 u}{\\partial y\\partial y}");
  ReplaceAll(&out, "\\partial_{zz}", "\\partial_{yy}");
  ReplaceAll(&out, "u_{zz}", "u_{yy}");
  ReplaceAll(&out, "z", "y");
  return out;
}

// Boundary condition helpers (delegating to shared utilities)
bool BuildBoundaryLatex(const BoundaryInput& input, std::string* latex, std::string* error) {
  const BoundaryLatexResult result = ::BuildBoundaryLatex(input);
  if (latex) {
    *latex = result.latex;
  }
  if (error) {
    *error = result.error;
  }
  return result.ok;
}

bool BuildBoundarySpec(const BoundaryInput& left, const BoundaryInput& right,
                       const BoundaryInput& bottom, const BoundaryInput& top,
                       const BoundaryInput& front, const BoundaryInput& back,
                       std::string* spec, std::string* error);

bool SetBoundaryFromSpec(const std::string& spec, BoundaryInput* bc) {
  if (!bc) {
    return false;
  }
  const std::string trimmed = Trim(spec);
  if (trimmed.empty()) {
    return false;
  }
  const size_t colon = trimmed.find(':');
  if (colon == std::string::npos) {
    bc->kind = 0;
    bc->value = trimmed;
    return true;
  }
  const std::string kind = ToLower(Trim(trimmed.substr(0, colon)));
  const std::string rest = Trim(trimmed.substr(colon + 1));
  if (kind == "dirichlet") {
    bc->kind = 0;
    bc->value = rest.empty() ? "0" : rest;
    return true;
  }
  if (kind == "neumann") {
    bc->kind = 1;
    bc->value = rest.empty() ? "0" : rest;
    return true;
  }
  if (kind == "robin") {
    bc->kind = 2;
    std::string alpha = bc->alpha.empty() ? "1" : bc->alpha;
    std::string beta = bc->beta.empty() ? "1" : bc->beta;
    std::string gamma = bc->gamma.empty() ? "0" : bc->gamma;
    const std::vector<std::string> parts = Split(rest, ',');
    for (const std::string& part : parts) {
      const size_t eq = part.find('=');
      if (eq == std::string::npos) {
        continue;
      }
      const std::string key = ToLower(Trim(part.substr(0, eq)));
      const std::string value = Trim(part.substr(eq + 1));
      if (value.empty()) {
        continue;
      }
      if (key == "alpha") {
        alpha = value;
      } else if (key == "beta") {
        beta = value;
      } else if (key == "gamma") {
        gamma = value;
      }
    }
    bc->alpha = alpha;
    bc->beta = beta;
    bc->gamma = gamma;
    return true;
  }
  return false;
}


// Process execution
bool RunProcess(const std::vector<std::string>& args, std::string* output) {
  if (args.empty()) {
    if (output) {
      *output = "missing process args";
    }
    return false;
  }

  int pipefd[2];
  if (pipe(pipefd) != 0) {
    if (output) {
      *output = "failed to open pipe";
    }
    return false;
  }

  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
  posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDERR_FILENO);
  posix_spawn_file_actions_addclose(&actions, pipefd[0]);

  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (const std::string& arg : args) {
    argv.push_back(const_cast<char*>(arg.c_str()));
  }
  argv.push_back(nullptr);

  pid_t pid = 0;
  int status = posix_spawnp(&pid, argv[0], &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  close(pipefd[1]);

  std::string buffer;
  if (status == 0) {
    char temp[256];
    ssize_t count = 0;
    while ((count = read(pipefd[0], temp, sizeof(temp))) > 0) {
      buffer.append(temp, static_cast<size_t>(count));
    }
  }
  close(pipefd[0]);

  if (status != 0) {
    if (output) {
      *output = "failed to start renderer";
    }
    return false;
  }

  int exit_status = 0;
  waitpid(pid, &exit_status, 0);
  const bool ok = WIFEXITED(exit_status) && WEXITSTATUS(exit_status) == 0;
  if (!ok && output) {
    *output = buffer.empty() ? "render failed" : Trim(buffer);
  }
  return ok;
}

bool RenderLatexToPng(const std::string& python, const std::filesystem::path& script,
                      const std::string& latex, const std::filesystem::path& out_path,
                      const std::string& color, int font_size, std::string* error) {
  if (latex.empty()) {
    if (error) {
      *error = "empty latex";
    }
    return false;
  }
  std::vector<std::string> args;
  args.push_back(python);
  args.push_back(script.string());
  args.push_back("--latex");
  args.push_back(latex);
  args.push_back("--out");
  args.push_back(out_path.string());
  args.push_back("--color");
  args.push_back(color);
  args.push_back("--fontsize");
  args.push_back(std::to_string(std::max(8, font_size)));
  return RunProcess(args, error);
}

bool LoadTextureFromFile(const std::filesystem::path& path, unsigned int* texture, int* width,
                         int* height, std::string* error) {
  int w = 0;
  int h = 0;
  int channels = 0;
  stbi_uc* data = stbi_load(path.string().c_str(), &w, &h, &channels, 4);
  if (!data) {
    if (error) {
      *error = stbi_failure_reason() ? stbi_failure_reason() : "failed to load png";
    }
    return false;
  }

  if (*texture == 0) {
    glGenTextures(1, reinterpret_cast<GLuint*>(texture));
  }
  glBindTexture(GL_TEXTURE_2D, *reinterpret_cast<GLuint*>(texture));
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);

  stbi_image_free(data);
  if (width) {
    *width = w;
  }
  if (height) {
    *height = h;
  }
  return true;
}

void UpdateLatexTexture(LatexTexture& tex, const std::string& source,
                        const std::string& python_path, const std::filesystem::path& script_path,
                        const std::filesystem::path& cache_dir,
                        const std::string& color, int font_size) {
  if (script_path.empty()) {
    if (!source.empty()) {
      tex.error = "latex renderer not found";
    }
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  if (source != tex.source || color != tex.color || font_size != tex.font_size) {
    tex.source = source;
    tex.color = color;
    tex.font_size = font_size;
    tex.dirty = true;
    tex.last_edit = now;
  }
  if (!tex.dirty) {
    return;
  }
  if (now - tex.last_edit < std::chrono::milliseconds(350)) {
    return;
  }

  tex.dirty = false;
  tex.error.clear();
  tex.last_rendered = tex.source;

  if (tex.source.empty()) {
    return;
  }

  std::hash<std::string> hasher;
  const size_t tag = hasher(tex.source + color + std::to_string(tex.font_size));
  std::filesystem::path out_path = cache_dir / ("latex_" + std::to_string(tag) + ".png");

  std::string render_error;
  if (!RenderLatexToPng(python_path, script_path, tex.source, out_path, color, tex.font_size,
                        &render_error)) {
    tex.error = render_error.empty() ? "render failed" : render_error;
    return;
  }

  std::string load_error;
  if (!LoadTextureFromFile(out_path, &tex.texture, &tex.width, &tex.height, &load_error)) {
    tex.error = load_error.empty() ? "failed to load preview" : load_error;
  }
}


// Backend and method conversion
BackendKind BackendFromIndex(int index) {
  switch (index) {
    case 1:
      return BackendKind::CPU;
    case 2:
      return BackendKind::CUDA;
    case 3:
      return BackendKind::Metal;
    case 4:
      return BackendKind::TPU;
    default:
      return BackendKind::Auto;
  }
}

int BackendToIndex(BackendKind kind) {
  switch (kind) {
    case BackendKind::CPU:
      return 1;
    case BackendKind::CUDA:
      return 2;
    case BackendKind::Metal:
      return 3;
    case BackendKind::TPU:
      return 4;
    case BackendKind::Auto:
    default:
      return 0;
  }
}

SolveMethod MethodFromIndex(int index) {
  switch (index) {
    case 1:
      return SolveMethod::GaussSeidel;
    case 2:
      return SolveMethod::SOR;
    case 3:
      return SolveMethod::CG;
    case 4:
      return SolveMethod::BiCGStab;
    case 5:
      return SolveMethod::GMRES;
    case 6:
      return SolveMethod::MultigridVcycle;
    case 0:
    default:
      return SolveMethod::Jacobi;
  }
}

int MethodToIndex(SolveMethod method) {
  switch (method) {
    case SolveMethod::GaussSeidel:
      return 1;
    case SolveMethod::SOR:
      return 2;
    case SolveMethod::CG:
      return 3;
    case SolveMethod::BiCGStab:
      return 4;
    case SolveMethod::GMRES:
      return 5;
    case SolveMethod::MultigridVcycle:
      return 6;
    case SolveMethod::Jacobi:
    default:
      return 0;
  }
}

// Drawing helpers
struct Vec3 {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

Vec3 RotateVec(const Vec3& v, float yaw, float pitch) {
  const float cx = std::cos(pitch);
  const float sx = std::sin(pitch);
  const float cy = std::cos(yaw);
  const float sy = std::sin(yaw);

  Vec3 out;
  const float x1 = v.x;
  const float y1 = v.y * cx - v.z * sx;
  const float z1 = v.y * sx + v.z * cx;
  out.x = x1 * cy + z1 * sy;
  out.y = y1;
  out.z = -x1 * sy + z1 * cy;
  return out;
}

ImU32 AxisColor(float r, float g, float b, float depth) {
  const float shade = 0.4f + 0.6f * std::max(0.0f, std::min(1.0f, (depth + 1.0f) * 0.5f));
  return ImColor(r * shade, g * shade, b * shade, 1.0f);
}

void DrawLatexPreview(const LatexTexture& tex, float max_width, float max_height) {
  if (tex.texture == 0) {
    return;
  }
  float w = static_cast<float>(tex.width);
  float h = static_cast<float>(tex.height);
  if (w <= 0.0f || h <= 0.0f) {
    w = max_width;
    h = max_height;
  }
  float scale = 1.0f;
  if (w > max_width) {
    scale = max_width / w;
  }
  if (h * scale > max_height) {
    scale = std::min(scale, max_height / h);
  }
  ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(tex.texture)),
               ImVec2(w * scale, h * scale), ImVec2(0, 0), ImVec2(1, 1));
}

void DrawGimbal(GlViewer& viewer, const ImVec2& top_right, float size, ImGuiIO& io) {
  const ImVec2 pos(top_right.x - size, top_right.y);
  const ImVec2 size_vec(size, size);
  ImGui::SetCursorScreenPos(pos);
  ImGui::InvisibleButton("gimbal", size_vec);

  if (ImGui::IsItemActive()) {
    viewer.Rotate(io.MouseDelta.x, io.MouseDelta.y);
  }

  float yaw = 0.0f;
  float pitch = 0.0f;
  viewer.GetOrientation(&yaw, &pitch);

  const ImVec2 center(pos.x + size * 0.5f, pos.y + size * 0.5f);
  const float radius = size * 0.42f;

  ImDrawList* draw = ImGui::GetWindowDrawList();
  draw->AddCircle(center, radius, IM_COL32(220, 220, 220, 200), 32, 1.5f);

  Vec3 axis_x = RotateVec({1.0f, 0.0f, 0.0f}, yaw, pitch);
  Vec3 axis_y = RotateVec({0.0f, 1.0f, 0.0f}, yaw, pitch);
  Vec3 axis_z = RotateVec({0.0f, 0.0f, 1.0f}, yaw, pitch);

  ImVec2 end_x(center.x + axis_x.x * radius, center.y - axis_x.y * radius);
  ImVec2 end_y(center.x + axis_y.x * radius, center.y - axis_y.y * radius);
  ImVec2 end_z(center.x + axis_z.x * radius, center.y - axis_z.y * radius);

  draw->AddLine(center, end_x, AxisColor(0.95f, 0.25f, 0.25f, axis_x.z), 2.0f);
  draw->AddLine(center, end_y, AxisColor(0.25f, 0.9f, 0.35f, axis_y.z), 2.0f);
  draw->AddLine(center, end_z, AxisColor(0.3f, 0.6f, 1.0f, axis_z.z), 2.0f);
  draw->AddCircleFilled(center, 2.5f, IM_COL32(230, 230, 230, 220));
}

void DrawGimbalForeground(GlViewer& viewer, const ImVec2& top_right, float size, ImGuiIO& io) {
  // Position the gimbal - top_right is the top-right corner reference point
  const ImVec2 pos(top_right.x - size, top_right.y);
  const ImVec2 size_vec(size, size);

  // Create invisible button for interaction
  ImGui::SetCursorScreenPos(pos);
  ImGui::InvisibleButton("gimbal_fg", size_vec);

  if (ImGui::IsItemActive()) {
    viewer.Rotate(io.MouseDelta.x, io.MouseDelta.y);
  }

  float yaw = 0.0f;
  float pitch = 0.0f;
  viewer.GetOrientation(&yaw, &pitch);

  const ImVec2 center(pos.x + size * 0.5f, pos.y + size * 0.5f);
  const float radius = size * 0.42f;

  // Use foreground draw list to ensure gimbal draws on top of all windows
  ImDrawList* draw = ImGui::GetForegroundDrawList();

  // Draw background circle for visibility
  draw->AddCircleFilled(center, radius + 4.0f, IM_COL32(30, 30, 35, 180));
  draw->AddCircle(center, radius, IM_COL32(220, 220, 220, 200), 32, 1.5f);

  Vec3 axis_x = RotateVec({1.0f, 0.0f, 0.0f}, yaw, pitch);
  Vec3 axis_y = RotateVec({0.0f, 1.0f, 0.0f}, yaw, pitch);
  Vec3 axis_z = RotateVec({0.0f, 0.0f, 1.0f}, yaw, pitch);

  ImVec2 end_x(center.x + axis_x.x * radius, center.y - axis_x.y * radius);
  ImVec2 end_y(center.x + axis_y.x * radius, center.y - axis_y.y * radius);
  ImVec2 end_z(center.x + axis_z.x * radius, center.y - axis_z.y * radius);

  draw->AddLine(center, end_x, AxisColor(0.95f, 0.25f, 0.25f, axis_x.z), 2.5f);
  draw->AddLine(center, end_y, AxisColor(0.25f, 0.9f, 0.35f, axis_y.z), 2.5f);
  draw->AddLine(center, end_z, AxisColor(0.3f, 0.6f, 1.0f, axis_z.z), 2.5f);
  draw->AddCircleFilled(center, 3.0f, IM_COL32(230, 230, 230, 220));
}

void DrawAxisLabels(const GlViewer& viewer, const ImVec2& image_min,
                    const ImVec2& image_max, const ImVec4& color) {
  const std::vector<GlViewer::ScreenLabel> labels = viewer.AxisLabels();
  if (labels.empty()) {
    return;
  }
  if (image_max.x <= image_min.x || image_max.y <= image_min.y) {
    return;
  }

  struct LabelBounds {
    float x0;
    float y0;
    float x1;
    float y1;
  };
  auto overlaps = [](const LabelBounds& a, const LabelBounds& b) {
    return !(a.x1 < b.x0 || a.x0 > b.x1 || a.y1 < b.y0 || a.y0 > b.y1);
  };

  ImDrawList* draw = ImGui::GetWindowDrawList();
  const ImU32 text_color = ImGui::ColorConvertFloat4ToU32(color);
  const float pad = 6.0f;
  const float box_pad = 2.0f;
  std::vector<LabelBounds> placed;
  placed.reserve(labels.size());

  for (const auto& label : labels) {
    if (label.text.empty()) {
      continue;
    }
    float x = image_min.x + label.x;
    float y = image_min.y + label.y;
    if (x < image_min.x || x > image_max.x || y < image_min.y || y > image_max.y) {
      continue;
    }
    const ImVec2 text_size = ImGui::CalcTextSize(label.text.c_str());
    if (text_size.x <= 0.0f || text_size.y <= 0.0f) {
      continue;
    }
    float text_x = label.align_right ? (x - text_size.x - pad) : (x + pad);
    float text_y = y - text_size.y * 0.5f;
    const float max_x = image_max.x - text_size.x - 1.0f;
    const float max_y = image_max.y - text_size.y - 1.0f;
    if (max_x < image_min.x + 1.0f || max_y < image_min.y + 1.0f) {
      continue;
    }
    text_x = std::min(std::max(text_x, image_min.x + 1.0f), max_x);
    text_y = std::min(std::max(text_y, image_min.y + 1.0f), max_y);
    LabelBounds bounds{
        text_x - box_pad,
        text_y - box_pad,
        text_x + text_size.x + box_pad,
        text_y + text_size.y + box_pad};
    bool hit = false;
    for (const auto& placed_bounds : placed) {
      if (overlaps(bounds, placed_bounds)) {
        hit = true;
        break;
      }
    }
    if (hit) {
      continue;
    }
    draw->AddText(ImVec2(text_x, text_y), text_color, label.text.c_str());
    placed.push_back(bounds);
  }
}
