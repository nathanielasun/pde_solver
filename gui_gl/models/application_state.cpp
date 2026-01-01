#include "application_state.h"
#include "utils/coordinate_utils.h"
#include <fstream>
#include <sstream>
#include <thread>
#include <algorithm>
#include <unistd.h>

// Simple JSON-like serialization (basic implementation)
// For production, consider using nlohmann/json or similar

namespace {
  std::string EscapeString(const std::string& str) {
    std::string result;
    result.reserve(str.size() + 10);
    for (char c : str) {
      if (c == '"' || c == '\\' || c == '\n' || c == '\r' || c == '\t') {
        result += '\\';
        if (c == '\n') result += 'n';
        else if (c == '\r') result += 'r';
        else if (c == '\t') result += 't';
        else result += c;
      } else {
        result += c;
      }
    }
    return result;
  }
  
  std::string UnescapeString(const std::string& str) {
    std::string result;
    result.reserve(str.size());
    for (size_t i = 0; i < str.size(); ++i) {
      if (str[i] == '\\' && i + 1 < str.size()) {
        if (str[i + 1] == 'n') { result += '\n'; ++i; }
        else if (str[i + 1] == 'r') { result += '\r'; ++i; }
        else if (str[i + 1] == 't') { result += '\t'; ++i; }
        else { result += str[i + 1]; ++i; }
      } else {
        result += str[i];
      }
    }
    return result;
  }
}

ApplicationState::ApplicationState() {
  // Initialize default state
  unsigned int hw = std::thread::hardware_concurrency();
  if (hw > 0) {
    state_.max_threads = static_cast<int>(hw);
  } else {
#ifdef _SC_NPROCESSORS_ONLN
    const long count = sysconf(_SC_NPROCESSORS_ONLN);
    state_.max_threads = count > 0 ? static_cast<int>(count) : 1;
#else
    state_.max_threads = 1;
#endif
  }
  state_.thread_count = state_.max_threads;
}

ApplicationState::State ApplicationState::GetState() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return state_;
}

ApplicationState::State& ApplicationState::GetMutableState() {
  return state_;  // Caller must ensure thread safety
}

void ApplicationState::SetState(const State& new_state) {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_ = new_state;
  }
  NotifyObservers();
}

void ApplicationState::SetPDEText(const std::string& text) {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.pde_text = text;
  }
  NotifyObservers();
}

void ApplicationState::SetDomain(const Domain& domain) {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.domain = domain;
  }
  NotifyObservers();
}

void ApplicationState::SetCoordinateMode(int mode) {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.coord_mode = mode;
  }
  UpdateCoordinateFlags();
  NotifyObservers();
}

void ApplicationState::SetBoundaryCondition(const std::string& face, const BoundaryInput& bc) {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (face == "left") state_.bc_left = bc;
    else if (face == "right") state_.bc_right = bc;
    else if (face == "bottom") state_.bc_bottom = bc;
    else if (face == "top") state_.bc_top = bc;
    else if (face == "front") state_.bc_front = bc;
    else if (face == "back") state_.bc_back = bc;
  }
  NotifyObservers();
}

void ApplicationState::SetSolverConfig(const SolverConfig& config) {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.solver = config;
  }
  NotifyObservers();
}

void ApplicationState::AddObserver(StateObserver observer) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  observers_.push_back(observer);
}

void ApplicationState::RemoveObserver(StateObserver observer) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  // Note: This is a simplified implementation. In production, you'd want
  // to use a more sophisticated observer management system.
  observers_.erase(
    std::remove_if(observers_.begin(), observers_.end(),
                   [&observer](const StateObserver& obs) {
                     // Compare function pointers (simplified)
                     return &obs == &observer;
                   }),
    observers_.end()
  );
}

bool ApplicationState::SaveToFile(const std::filesystem::path& path) const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  
  std::ofstream file(path);
  if (!file.is_open()) {
    return false;
  }
  
  // Simple JSON-like format (basic implementation)
  file << "{\n";
  file << "  \"pde_text\": \"" << EscapeString(state_.pde_text) << "\",\n";
  file << "  \"coord_mode\": " << state_.coord_mode << ",\n";
  file << "  \"domain_mode\": " << state_.domain_mode << ",\n";
  file << "  \"domain_shape\": \"" << EscapeString(state_.domain_shape) << "\",\n";
  file << "  \"domain_shape_file\": \"" << EscapeString(state_.domain_shape_file) << "\",\n";
  file << "  \"domain_shape_mask\": \"" << EscapeString(state_.domain_shape_mask) << "\",\n";
  file << "  \"domain_shape_mask_threshold\": " << state_.domain_shape_mask_threshold << ",\n";
  file << "  \"domain_shape_mask_invert\": " << (state_.domain_shape_mask_invert ? "true" : "false") << ",\n";
  file << "  \"shape_offset\": [" << state_.shape_transform.offset_x << ", "
       << state_.shape_transform.offset_y << ", " << state_.shape_transform.offset_z << "],\n";
  file << "  \"shape_scale\": [" << state_.shape_transform.scale_x << ", "
       << state_.shape_transform.scale_y << ", " << state_.shape_transform.scale_z << "],\n";
  file << "  \"bound_xmin\": " << state_.bound_xmin << ",\n";
  file << "  \"bound_xmax\": " << state_.bound_xmax << ",\n";
  file << "  \"bound_ymin\": " << state_.bound_ymin << ",\n";
  file << "  \"bound_ymax\": " << state_.bound_ymax << ",\n";
  file << "  \"bound_zmin\": " << state_.bound_zmin << ",\n";
  file << "  \"bound_zmax\": " << state_.bound_zmax << ",\n";
  file << "  \"grid_nx\": " << state_.grid_nx << ",\n";
  file << "  \"grid_ny\": " << state_.grid_ny << ",\n";
  file << "  \"grid_nz\": " << state_.grid_nz << ",\n";
  file << "  \"sor_omega\": " << state_.sor_omega << ",\n";
  file << "  \"gmres_restart\": " << state_.gmres_restart << ",\n";
  file << "  \"method_index\": " << state_.method_index << ",\n";
  file << "  \"backend_index\": " << state_.backend_index << "\n";
  file << "}\n";
  
  return file.good();
}

bool ApplicationState::LoadFromFile(const std::filesystem::path& path) {
  // Basic implementation - in production, use a proper JSON parser
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }
  
  // Simple parser (very basic - for production use nlohmann/json)
  std::string line;
  State new_state = state_;  // Start with current state
  
  while (std::getline(file, line)) {
    // Very basic parsing - just extract key-value pairs
    // This is a placeholder - real implementation would use proper JSON parsing
    if (line.find("\"pde_text\"") != std::string::npos) {
      // Extract value (simplified)
      size_t start = line.find('"', line.find(':'));
      size_t end = line.rfind('"');
      if (start != std::string::npos && end != std::string::npos && end > start) {
        new_state.pde_text = UnescapeString(line.substr(start + 1, end - start - 1));
      }
    }
    // Add more field parsing as needed
  }
  
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_ = new_state;
  }
  UpdateCoordinateFlags();
  NotifyObservers();
  
  return true;
}

void ApplicationState::NotifyStateChanged() {
  NotifyObservers();
}

void ApplicationState::UpdateCoordinateFlags() {
  std::lock_guard<std::mutex> lock(state_mutex_);
  
  // Reset all flags
  state_.use_polar_coords = false;
  state_.use_cartesian_3d = false;
  state_.use_axisymmetric = false;
  state_.use_cylindrical_volume = false;
  state_.use_spherical_surface = false;
  state_.use_spherical_volume = false;
  state_.use_toroidal_surface = false;
  state_.use_toroidal_volume = false;
  state_.use_surface = false;
  state_.use_volume = false;
  
  // Set flags based on coord_mode
  using namespace CoordMode;
  switch (state_.coord_mode) {
    case kCartesian2D:
      // Default flags
      break;
    case kCartesian3D:
      state_.use_cartesian_3d = true;
      state_.use_volume = true;
      break;
    case kPolar:
      state_.use_polar_coords = true;
      break;
    case kAxisymmetric:
      state_.use_axisymmetric = true;
      break;
    case kCylindricalVolume:
      state_.use_cylindrical_volume = true;
      state_.use_volume = true;
      break;
    case kSphericalSurface:
      state_.use_spherical_surface = true;
      state_.use_surface = true;
      break;
    case kSphericalVolume:
      state_.use_spherical_volume = true;
      state_.use_volume = true;
      break;
    case kToroidalSurface:
      state_.use_toroidal_surface = true;
      state_.use_surface = true;
      break;
    case kToroidalVolume:
      state_.use_toroidal_volume = true;
      state_.use_volume = true;
      break;
  }
}

void ApplicationState::NotifyObservers() const {
  State current_state;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    current_state = state_;
  }
  
  // Notify all observers (without holding the lock)
  for (const auto& observer : observers_) {
    if (observer) {
      observer(current_state);
    }
  }
}
