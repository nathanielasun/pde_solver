#ifndef IMAGE_EXPORT_H
#define IMAGE_EXPORT_H

#include "GlViewer.h"
#include "pde_types.h"
#include "components/inspection_tools.h"
#include <string>
#include <vector>
#include <filesystem>

// Image export utility for capturing OpenGL framebuffer with axis labels
namespace ImageExport {

// Export options
struct ExportOptions {
  int width = 1920;           // Export width
  int height = 1080;           // Export height
  bool include_axis_labels = true;  // Include coordinate axis labels
  bool include_colorbar = false;    // Include colorbar (future feature)
  int jpeg_quality = 95;      // JPEG quality (1-100)
};

// Export the current viewer state to an image file
// Returns true on success, false on failure
bool ExportImage(GlViewer& viewer, const std::filesystem::path& filepath, 
                 const ExportOptions& options = ExportOptions());

// Get suggested filename based on current state
std::string GetSuggestedFilename(const GlViewer& viewer, 
                                  const std::string& extension = "png");

// Line plot export options
struct LinePlotExportOptions {
  int width = 1920;           // Export width
  int height = 1080;          // Export height
  bool include_grid = true;  // Include coordinate grid
  bool include_axis_labels = true;  // Include axis labels
  bool include_metadata = true;  // Include metadata text at top
  int jpeg_quality = 95;     // JPEG quality (1-100)
  int metadata_height = 120; // Height reserved for metadata text at top
};

// Export a line plot as a 2D image with coordinate grid, axis labels, and metadata
// plot: The line plot data (positions and values)
// domain: Domain information for coordinate labels
// field_type: Current field type being visualized
// Returns true on success, false on failure
bool ExportLinePlotImage(const LinePlot& plot, const Domain& domain,
                         GlViewer::FieldType field_type,
                         const std::filesystem::path& filepath,
                         const LinePlotExportOptions& options = LinePlotExportOptions());

// Get suggested filename for line plot export
std::string GetLinePlotFilename(const LinePlot& plot,
                                 const std::string& extension = "png");

}  // namespace ImageExport

#endif  // IMAGE_EXPORT_H

