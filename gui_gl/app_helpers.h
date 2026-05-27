#ifndef APP_HELPERS_H
#define APP_HELPERS_H

#include "GlViewer.h"
#include "app_state.h"
#include "utils/string_utils.h"
#include "utils/math_utils.h"
#include "utils/coordinate_utils.h"
#include "utils/path_utils.h"
#include "io/preferences_io.h"
#include "io/file_utils.h"
#include <string>
#include <vector>
#include <filesystem>
#include <optional>
#include <chrono>

#include "latex/latex_texture.h"
#include "latex/latex_preview_draw.h"

// Forward declarations
struct BoundaryInput;
struct Preferences;
struct SharedState;
struct ImVec2;
struct ImGuiIO;

// Expression remapping for coordinate systems
std::string RemapSphericalSurfaceExpr(const std::string& input);
std::string RemapAxisymmetricExpr(const std::string& input);
// Boundary condition helpers
bool BuildBoundaryLatex(const BoundaryInput& input, std::string* latex, std::string* error);
bool BuildBoundarySpec(const BoundaryInput& left, const BoundaryInput& right,
                       const BoundaryInput& bottom, const BoundaryInput& top,
                       const BoundaryInput& front, const BoundaryInput& back,
                       std::string* spec, std::string* error);
bool SetBoundaryFromSpec(const std::string& spec, BoundaryInput* bc);

// LaTeX rendering (async in-process; see latex/latex_render_service.h)
#include "latex/latex_render_service.h"

bool RenderLatexToPng(const std::string& python, const std::filesystem::path& script,
                     const std::string& latex, const std::filesystem::path& out_path,
                     const std::string& color, int font_size, std::string* error);

// Process execution
bool RunProcess(const std::vector<std::string>& args, std::string* output);

// Backend and method conversion
#include "backend.h"
#include "pde_types.h"
BackendKind BackendFromIndex(int index);
int BackendToIndex(BackendKind kind);
SolveMethod MethodFromIndex(int index);
int MethodToIndex(SolveMethod method);

// Drawing helpers
void DrawGimbal(GlViewer& viewer, const ImVec2& top_right, float size, ImGuiIO& io);
void DrawGimbalForeground(GlViewer& viewer, const ImVec2& top_right, float size, ImGuiIO& io);
void DrawAxisLabels(const GlViewer& viewer, const ImVec2& image_min,
                    const ImVec2& image_max, const ImVec4& color);

#endif // APP_HELPERS_H
