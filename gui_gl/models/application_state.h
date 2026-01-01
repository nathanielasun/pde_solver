#ifndef APPLICATION_STATE_H
#define APPLICATION_STATE_H

#include "app_state.h"
#include "pde_types.h"
#include <string>
#include <vector>
#include <functional>
#include <mutex>
#include <filesystem>
#include <optional>

// Forward declarations
struct BoundaryInput;
struct DomainPanelState;
struct EquationPanelState;
struct GridPanelState;
struct BoundaryPanelState;
struct ComputePanelState;

/**
 * ApplicationState - Centralized state management for the PDE solver GUI
 * 
 * This class provides:
 * - Centralized storage of all UI state
 * - Observer pattern for state change notifications
 * - Serialization (save/load) capabilities
 * - Thread-safe state access
 */
class ApplicationState {
 public:
  // Main application state structure
  struct State {
    // PDE equation
    std::string pde_text = "-\\frac{\\partial^2 u}{\\partial x^2} = u";
    
    // Domain configuration
    Domain domain;
    int coord_mode = 0;  // Coordinate system mode index
    int domain_mode = 0;  // 0 = rectangular, 1 = implicit
    std::string domain_shape;  // Implicit domain shape function
    std::string domain_shape_file;
    std::string domain_shape_mask;
    double domain_shape_mask_threshold = 0.0;
    bool domain_shape_mask_invert = false;
    ShapeTransform shape_transform;
    
    // Domain bounds
    double bound_xmin = 0.0;
    double bound_xmax = 1.0;
    double bound_ymin = 0.0;
    double bound_ymax = 1.0;
    double bound_zmin = 0.0;
    double bound_zmax = 1.0;
    
    // Torus geometry (for toroidal coordinates)
    double torus_major = 1.6;
    double torus_minor = 0.45;
    
    // Grid configuration
    int grid_nx = 64;
    int grid_ny = 64;
    int grid_nz = 64;
    
    // Boundary conditions
    BoundaryInput bc_left;
    BoundaryInput bc_right;
    BoundaryInput bc_bottom;
    BoundaryInput bc_top;
    BoundaryInput bc_front;
    BoundaryInput bc_back;
    
    // Solver configuration
    SolverConfig solver;
    int backend_index = 0;
    int method_index = 0;
    double sor_omega = 1.5;
    int gmres_restart = 30;
    int thread_count = 0;
    int max_threads = 1;
    
    // Metal tuning
    int metal_reduce_interval = 10;
    int metal_tg_x = 0;
    int metal_tg_y = 0;
    
    // Time configuration
    double time_start = 0.0;
    double time_end = 1.0;
    int time_frames = 60;
    
    // Preferences
    Preferences preferences;
    
    // Preference values (for saving)
    int pref_method_index = 0;
    double pref_sor_omega = 1.5;
    int pref_gmres_restart = 30;
    int pref_metal_reduce_interval = 10;
    int pref_metal_tg_x = 0;
    int pref_metal_tg_y = 0;
    
    // Coordinate flags (computed, but stored for convenience)
    bool use_polar_coords = false;
    bool use_cartesian_3d = false;
    bool use_axisymmetric = false;
    bool use_cylindrical_volume = false;
    bool use_spherical_surface = false;
    bool use_spherical_volume = false;
    bool use_toroidal_surface = false;
    bool use_toroidal_volume = false;
    bool use_surface = false;
    bool use_volume = false;
    
    // Inspection tools state
    bool slice_enabled = false;
    int slice_axis = 0;
    double slice_value = 0.0;
    double slice_thickness = 0.0;
    bool iso_enabled = false;
    double iso_value = 0.0;
    double iso_band = 0.0;
    int field_type_index = 0;
    
    // File paths
    std::vector<std::filesystem::path> frame_paths;
    std::vector<double> frame_times;
    int frame_index = 0;
    int last_loaded_frame = -1;
    bool playing = false;
    std::string output_path = "outputs";
    std::string input_dir = "outputs";
    
    // Shared state (solver results, progress, etc.)
    SharedState shared_state;
  };
  
  ApplicationState();
  ~ApplicationState() = default;
  
  // State access (thread-safe)
  State GetState() const;
  State& GetMutableState();  // Non-const access (use with caution)
  void SetState(const State& new_state);
  
  // Individual field setters (with notifications)
  void SetPDEText(const std::string& text);
  void SetDomain(const Domain& domain);
  void SetCoordinateMode(int mode);
  void SetBoundaryCondition(const std::string& face, const BoundaryInput& bc);
  void SetSolverConfig(const SolverConfig& config);
  
  // Observer pattern
  using StateObserver = std::function<void(const State&)>;
  void AddObserver(StateObserver observer);
  void RemoveObserver(StateObserver observer);
  
  // Serialization
  bool SaveToFile(const std::filesystem::path& path) const;
  bool LoadFromFile(const std::filesystem::path& path);
  
  // Convenience methods for common operations
  void NotifyStateChanged();
  void UpdateCoordinateFlags();  // Recompute coordinate flags from coord_mode
  
 private:
  mutable std::mutex state_mutex_;
  State state_;
  std::vector<StateObserver> observers_;
  
  void NotifyObservers() const;
};

#endif  // APPLICATION_STATE_H
