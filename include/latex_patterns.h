#ifndef LATEX_PATTERNS_H
#define LATEX_PATTERNS_H

// Central registry of recognized LaTeX patterns for PDE term parsing.
// Kept header-only (C++17 inline variables) to avoid additional compilation units.

namespace LatexPatterns {

inline constexpr const char* kD2XPatterns[] = {
    "\\frac{\\partial^2u}{\\partialx^2}",
    "\\frac{\\partial^2u}{\\partialx\\partialx}",
    "\\partial_{xx}u",
    "\\partial_x\\partial_xu",
    "u_{xx}",
    "\\partial^2u/\\partialx^2",
    "\\partial^2u/\\partialx\\partialx",
    "\\frac{d^2u}{dx^2}",
    "\\frac{d^2u}{dxdx}",
    "d^2u/dx^2",
    "d^2u/dxdx",
    "\\frac{\\partial^2u}{\\partialr^2}",
    "\\frac{\\partial^2u}{\\partialr\\partialr}",
    "\\partial_{rr}u",
    "\\partial_r\\partial_ru",
    "u_{rr}",
    "\\partial^2u/\\partialr^2",
    "\\partial^2u/\\partialr\\partialr",
    "\\frac{d^2u}{dr^2}",
    "\\frac{d^2u}{drdr}",
    "d^2u/dr^2",
    "d^2u/drdr",
};

inline constexpr const char* kD2YPatterns[] = {
    "\\frac{\\partial^2u}{\\partialy^2}",
    "\\frac{\\partial^2u}{\\partialy\\partialy}",
    "\\partial_{yy}u",
    "\\partial_y\\partial_yu",
    "u_{yy}",
    "\\partial^2u/\\partialy^2",
    "\\partial^2u/\\partialy\\partialy",
    "\\frac{d^2u}{dy^2}",
    "\\frac{d^2u}{dydy}",
    "d^2u/dy^2",
    "d^2u/dydy",
    "\\frac{\\partial^2u}{\\partialtheta^2}",
    "\\frac{\\partial^2u}{\\partialtheta\\partialtheta}",
    "\\partial_{thetatheta}u",
    "\\partial_theta\\partial_thetau",
    "u_{thetatheta}",
    "\\partial^2u/\\partialtheta^2",
    "\\partial^2u/\\partialtheta\\partialtheta",
    "\\frac{d^2u}{dtheta^2}",
    "\\frac{d^2u}{dthetadtheta}",
    "d^2u/dtheta^2",
    "d^2u/dthetadtheta",
};

inline constexpr const char* kD2ZPatterns[] = {
    "\\frac{\\partial^2u}{\\partialz^2}",
    "\\frac{\\partial^2u}{\\partialz\\partialz}",
    "\\partial_{zz}u",
    "\\partial_z\\partial_zu",
    "u_{zz}",
    "\\partial^2u/\\partialz^2",
    "\\partial^2u/\\partialz\\partialz",
    "\\frac{d^2u}{dz^2}",
    "\\frac{d^2u}{dzdz}",
    "d^2u/dz^2",
    "d^2u/dzdz",
    "\\frac{\\partial^2u}{\\partialphi^2}",
    "\\frac{\\partial^2u}{\\partialphi\\partialphi}",
    "\\partial_{phiphi}u",
    "\\partial_phi\\partial_phiu",
    "u_{phiphi}",
    "\\partial^2u/\\partialphi^2",
    "\\partial^2u/\\partialphi\\partialphi",
    "\\frac{d^2u}{dphi^2}",
    "\\frac{d^2u}{dphidphi}",
    "d^2u/dphi^2",
    "d^2u/dphidphi",
};

inline constexpr const char* kDXPatterns[] = {
    "\\frac{\\partialu}{\\partialx}",
    "\\partial_xu",
    "u_x",
    "\\partialu/\\partialx",
    "\\frac{du}{dx}",
    "du/dx",
    "\\frac{\\partialu}{\\partialr}",
    "\\partial_ru",
    "u_r",
    "\\partialu/\\partialr",
    "\\frac{du}{dr}",
    "du/dr",
};

inline constexpr const char* kDYPatterns[] = {
    "\\frac{\\partialu}{\\partialy}",
    "\\partial_yu",
    "u_y",
    "\\partialu/\\partialy",
    "\\frac{du}{dy}",
    "du/dy",
    "\\frac{\\partialu}{\\partialtheta}",
    "\\partial_thetau",
    "u_theta",
    "\\partialu/\\partialtheta",
    "\\frac{du}{dtheta}",
    "du/dtheta",
};

inline constexpr const char* kDZPatterns[] = {
    "\\frac{\\partialu}{\\partialz}",
    "\\partial_zu",
    "u_z",
    "\\partialu/\\partialz",
    "\\frac{du}{dz}",
    "du/dz",
    "\\frac{\\partialu}{\\partialphi}",
    "\\partial_phiu",
    "u_phi",
    "\\partialu/\\partialphi",
    "\\frac{du}{dphi}",
    "du/dphi",
};

inline constexpr const char* kDTPatterns[] = {
    "\\frac{\\partialu}{\\partialt}",
    "\\partial_tu",
    "u_t",
    "\\partialu/\\partialt",
    "\\frac{du}{dt}",
    "du/dt",
    "\\dot{u}",
    "\\dotu",
};

inline constexpr const char* kD2TPatterns[] = {
    "\\frac{\\partial^2u}{\\partialt^2}",
    "\\frac{\\partial^2u}{\\partialt\\partialt}",
    "\\partial_{tt}u",
    "\\partial_t\\partial_tu",
    "u_{tt}",
    "\\partial^2u/\\partialt^2",
    "\\partial^2u/\\partialt\\partialt",
    "\\frac{d^2u}{dt^2}",
    "\\frac{d^2u}{dtdt}",
    "d^2u/dt^2",
    "d^2u/dtdt",
    "\\ddot{u}",
    "\\ddotu",
};

// Mixed derivative patterns: u_xy (and u_yx, symmetric)
inline constexpr const char* kDXYPatterns[] = {
    "u_{xy}",
    "u_{yx}",
    "\\partial_{xy}u",
    "\\partial_{yx}u",
    "\\partial_x\\partial_yu",
    "\\partial_y\\partial_xu",
    "\\frac{\\partial^2u}{\\partialx\\partialy}",
    "\\frac{\\partial^2u}{\\partialy\\partialx}",
    "\\partial^2u/\\partialx\\partialy",
    "\\partial^2u/\\partialy\\partialx",
    "\\frac{d^2u}{dxdy}",
    "\\frac{d^2u}{dydx}",
    "d^2u/dxdy",
    "d^2u/dydx",
    // Coordinate variants for polar/cylindrical
    "u_{rtheta}",
    "u_{thetar}",
    "\\partial_{rtheta}u",
    "\\partial_{thetar}u",
    "\\partial_r\\partial_thetau",
    "\\partial_theta\\partial_ru",
    "\\frac{\\partial^2u}{\\partialr\\partialtheta}",
    "\\frac{\\partial^2u}{\\partialtheta\\partialr}",
    "\\partial^2u/\\partialr\\partialtheta",
    "\\partial^2u/\\partialtheta\\partialr",
};

// Mixed derivative patterns: u_xz (and u_zx, symmetric)
inline constexpr const char* kDXZPatterns[] = {
    "u_{xz}",
    "u_{zx}",
    "\\partial_{xz}u",
    "\\partial_{zx}u",
    "\\partial_x\\partial_zu",
    "\\partial_z\\partial_xu",
    "\\frac{\\partial^2u}{\\partialx\\partialz}",
    "\\frac{\\partial^2u}{\\partialz\\partialx}",
    "\\partial^2u/\\partialx\\partialz",
    "\\partial^2u/\\partialz\\partialx",
    "\\frac{d^2u}{dxdz}",
    "\\frac{d^2u}{dzdx}",
    "d^2u/dxdz",
    "d^2u/dzdx",
    // Coordinate variants
    "u_{rphi}",
    "u_{phir}",
    "\\partial_{rphi}u",
    "\\partial_{phir}u",
    "\\partial_r\\partial_phiu",
    "\\partial_phi\\partial_ru",
    "\\frac{\\partial^2u}{\\partialr\\partialphi}",
    "\\frac{\\partial^2u}{\\partialphi\\partialr}",
    "\\partial^2u/\\partialr\\partialphi",
    "\\partial^2u/\\partialphi\\partialr",
};

// Mixed derivative patterns: u_yz (and u_zy, symmetric)
inline constexpr const char* kDYZPatterns[] = {
    "u_{yz}",
    "u_{zy}",
    "\\partial_{yz}u",
    "\\partial_{zy}u",
    "\\partial_y\\partial_zu",
    "\\partial_z\\partial_yu",
    "\\frac{\\partial^2u}{\\partialy\\partialz}",
    "\\frac{\\partial^2u}{\\partialz\\partialy}",
    "\\partial^2u/\\partialy\\partialz",
    "\\partial^2u/\\partialz\\partialy",
    "\\frac{d^2u}{dydz}",
    "\\frac{d^2u}{dzdy}",
    "d^2u/dydz",
    "d^2u/dzdy",
    // Coordinate variants
    "u_{thetaphi}",
    "u_{phitheta}",
    "\\partial_{thetaphi}u",
    "\\partial_{phitheta}u",
    "\\partial_theta\\partial_phiu",
    "\\partial_phi\\partial_thetau",
    "\\frac{\\partial^2u}{\\partialtheta\\partialphi}",
    "\\frac{\\partial^2u}{\\partialphi\\partialtheta}",
    "\\partial^2u/\\partialtheta\\partialphi",
    "\\partial^2u/\\partialphi\\partialtheta",
};

// Third derivative patterns: u_xxx
inline constexpr const char* kD3XPatterns[] = {
    "\\frac{\\partial^3u}{\\partialx^3}",
    "\\frac{\\partial^3u}{\\partialx\\partialx\\partialx}",
    "\\partial_{xxx}u",
    "\\partial_x\\partial_x\\partial_xu",
    "u_{xxx}",
    "\\partial^3u/\\partialx^3",
    "\\partial^3u/\\partialx\\partialx\\partialx",
    "\\frac{d^3u}{dx^3}",
    "\\frac{d^3u}{dxdxdx}",
    "d^3u/dx^3",
    "d^3u/dxdxdx",
    "\\frac{\\partial^3u}{\\partialr^3}",
    "\\partial_{rrr}u",
    "u_{rrr}",
};

// Third derivative patterns: u_yyy
inline constexpr const char* kD3YPatterns[] = {
    "\\frac{\\partial^3u}{\\partialy^3}",
    "\\frac{\\partial^3u}{\\partialy\\partialy\\partialy}",
    "\\partial_{yyy}u",
    "\\partial_y\\partial_y\\partial_yu",
    "u_{yyy}",
    "\\partial^3u/\\partialy^3",
    "\\partial^3u/\\partialy\\partialy\\partialy",
    "\\frac{d^3u}{dy^3}",
    "\\frac{d^3u}{dydydy}",
    "d^3u/dy^3",
    "d^3u/dydydy",
    "\\frac{\\partial^3u}{\\partialtheta^3}",
    "\\partial_{thetathetatheta}u",
    "u_{thetathetatheta}",
};

// Third derivative patterns: u_zzz
inline constexpr const char* kD3ZPatterns[] = {
    "\\frac{\\partial^3u}{\\partialz^3}",
    "\\frac{\\partial^3u}{\\partialz\\partialz\\partialz}",
    "\\partial_{zzz}u",
    "\\partial_z\\partial_z\\partial_zu",
    "u_{zzz}",
    "\\partial^3u/\\partialz^3",
    "\\partial^3u/\\partialz\\partialz\\partialz",
    "\\frac{d^3u}{dz^3}",
    "\\frac{d^3u}{dzdzdz}",
    "d^3u/dz^3",
    "d^3u/dzdzdz",
    "\\frac{\\partial^3u}{\\partialphi^3}",
    "\\partial_{phiphiphi}u",
    "u_{phiphiphi}",
};

// Fourth derivative patterns: u_xxxx
inline constexpr const char* kD4XPatterns[] = {
    "\\frac{\\partial^4u}{\\partialx^4}",
    "\\frac{\\partial^4u}{\\partialx\\partialx\\partialx\\partialx}",
    "\\partial_{xxxx}u",
    "\\partial_x\\partial_x\\partial_x\\partial_xu",
    "u_{xxxx}",
    "\\partial^4u/\\partialx^4",
    "\\partial^4u/\\partialx\\partialx\\partialx\\partialx",
    "\\frac{d^4u}{dx^4}",
    "\\frac{d^4u}{dxdxdxdx}",
    "d^4u/dx^4",
    "d^4u/dxdxdxdx",
    "\\frac{\\partial^4u}{\\partialr^4}",
    "\\partial_{rrrr}u",
    "u_{rrrr}",
};

// Fourth derivative patterns: u_yyyy
inline constexpr const char* kD4YPatterns[] = {
    "\\frac{\\partial^4u}{\\partialy^4}",
    "\\frac{\\partial^4u}{\\partialy\\partialy\\partialy\\partialy}",
    "\\partial_{yyyy}u",
    "\\partial_y\\partial_y\\partial_y\\partial_yu",
    "u_{yyyy}",
    "\\partial^4u/\\partialy^4",
    "\\partial^4u/\\partialy\\partialy\\partialy\\partialy",
    "\\frac{d^4u}{dy^4}",
    "\\frac{d^4u}{dydydydy}",
    "d^4u/dy^4",
    "d^4u/dydydydy",
    "\\frac{\\partial^4u}{\\partialtheta^4}",
    "\\partial_{thetathetathetatheta}u",
    "u_{thetathetathetatheta}",
};

// Fourth derivative patterns: u_zzzz
inline constexpr const char* kD4ZPatterns[] = {
    "\\frac{\\partial^4u}{\\partialz^4}",
    "\\frac{\\partial^4u}{\\partialz\\partialz\\partialz\\partialz}",
    "\\partial_{zzzz}u",
    "\\partial_z\\partial_z\\partial_z\\partial_zu",
    "u_{zzzz}",
    "\\partial^4u/\\partialz^4",
    "\\partial^4u/\\partialz\\partialz\\partialz\\partialz",
    "\\frac{d^4u}{dz^4}",
    "\\frac{d^4u}{dzdzdzdz}",
    "d^4u/dz^4",
    "d^4u/dzdzdzdz",
    "\\frac{\\partial^4u}{\\partialphi^4}",
    "\\partial_{phiphiphiphi}u",
    "u_{phiphiphiphi}",
};

inline constexpr const char* kLaplacianPatterns[] = {
    "\\nabla^2u",
    "\\Deltau",
    "\\triangleu",
};

inline constexpr const char* kU = "u";

// Reserved variable names (coordinates and time) that cannot be used as field names
inline bool IsReservedVariable(const std::string& name) {
  return name == "x" || name == "y" || name == "z" || name == "t" ||
         name == "r" || name == "theta" || name == "phi" || name == "rho";
}

// Valid field variable names: single lowercase letters or short alphanumeric names
// Excludes reserved coordinate/time variables
inline bool IsValidFieldName(const std::string& name) {
  if (name.empty() || name.length() > 10) {
    return false;
  }
  if (IsReservedVariable(name)) {
    return false;
  }
  // First character must be lowercase letter
  if (!std::isalpha(static_cast<unsigned char>(name[0])) ||
      !std::islower(static_cast<unsigned char>(name[0]))) {
    return false;
  }
  // Rest must be alphanumeric
  for (size_t i = 1; i < name.length(); ++i) {
    if (!std::isalnum(static_cast<unsigned char>(name[i]))) {
      return false;
    }
  }
  return true;
}

// Common field variable names for coupled PDE systems
inline const std::vector<std::string>& GetCommonFieldNames() {
  static const std::vector<std::string> names = {
    "u", "v", "w",           // Standard velocity/field components
    "p", "q",                // Pressure, additional fields
    "c", "s",                // Concentration, salinity
    "temp", "T",             // Temperature (note: T is uppercase, be careful)
  };
  return names;
}

// Pattern generators for multi-field support
// These generate derivative patterns for any field variable

// Generate second derivative patterns (field_xx, field_yy, etc.)
inline std::vector<std::string> GenerateD2XPatterns(const std::string& field) {
  return {
    "\\frac{\\partial^2" + field + "}{\\partialx^2}",
    "\\frac{\\partial^2" + field + "}{\\partialx\\partialx}",
    "\\partial_{xx}" + field,
    "\\partial_x\\partial_x" + field,
    field + "_{xx}",
    "\\partial^2" + field + "/\\partialx^2",
    "\\partial^2" + field + "/\\partialx\\partialx",
    "\\frac{d^2" + field + "}{dx^2}",
    "\\frac{d^2" + field + "}{dxdx}",
    "d^2" + field + "/dx^2",
    "d^2" + field + "/dxdx",
    // Radial variants
    "\\frac{\\partial^2" + field + "}{\\partialr^2}",
    "\\frac{\\partial^2" + field + "}{\\partialr\\partialr}",
    "\\partial_{rr}" + field,
    "\\partial_r\\partial_r" + field,
    field + "_{rr}",
    "\\partial^2" + field + "/\\partialr^2",
    "\\partial^2" + field + "/\\partialr\\partialr",
    "\\frac{d^2" + field + "}{dr^2}",
    "\\frac{d^2" + field + "}{drdr}",
    "d^2" + field + "/dr^2",
    "d^2" + field + "/drdr",
  };
}

inline std::vector<std::string> GenerateD2YPatterns(const std::string& field) {
  return {
    "\\frac{\\partial^2" + field + "}{\\partialy^2}",
    "\\frac{\\partial^2" + field + "}{\\partialy\\partialy}",
    "\\partial_{yy}" + field,
    "\\partial_y\\partial_y" + field,
    field + "_{yy}",
    "\\partial^2" + field + "/\\partialy^2",
    "\\partial^2" + field + "/\\partialy\\partialy",
    "\\frac{d^2" + field + "}{dy^2}",
    "\\frac{d^2" + field + "}{dydy}",
    "d^2" + field + "/dy^2",
    "d^2" + field + "/dydy",
    // Theta variants
    "\\frac{\\partial^2" + field + "}{\\partialtheta^2}",
    "\\frac{\\partial^2" + field + "}{\\partialtheta\\partialtheta}",
    "\\partial_{thetatheta}" + field,
    "\\partial_theta\\partial_theta" + field,
    field + "_{thetatheta}",
    "\\partial^2" + field + "/\\partialtheta^2",
    "\\partial^2" + field + "/\\partialtheta\\partialtheta",
    "\\frac{d^2" + field + "}{dtheta^2}",
    "\\frac{d^2" + field + "}{dthetadtheta}",
    "d^2" + field + "/dtheta^2",
    "d^2" + field + "/dthetadtheta",
  };
}

inline std::vector<std::string> GenerateD2ZPatterns(const std::string& field) {
  return {
    "\\frac{\\partial^2" + field + "}{\\partialz^2}",
    "\\frac{\\partial^2" + field + "}{\\partialz\\partialz}",
    "\\partial_{zz}" + field,
    "\\partial_z\\partial_z" + field,
    field + "_{zz}",
    "\\partial^2" + field + "/\\partialz^2",
    "\\partial^2" + field + "/\\partialz\\partialz",
    "\\frac{d^2" + field + "}{dz^2}",
    "\\frac{d^2" + field + "}{dzdz}",
    "d^2" + field + "/dz^2",
    "d^2" + field + "/dzdz",
    // Phi variants
    "\\frac{\\partial^2" + field + "}{\\partialphi^2}",
    "\\frac{\\partial^2" + field + "}{\\partialphi\\partialphi}",
    "\\partial_{phiphi}" + field,
    "\\partial_phi\\partial_phi" + field,
    field + "_{phiphi}",
    "\\partial^2" + field + "/\\partialphi^2",
    "\\partial^2" + field + "/\\partialphi\\partialphi",
    "\\frac{d^2" + field + "}{dphi^2}",
    "\\frac{d^2" + field + "}{dphidphi}",
    "d^2" + field + "/dphi^2",
    "d^2" + field + "/dphidphi",
  };
}

inline std::vector<std::string> GenerateDXPatterns(const std::string& field) {
  return {
    "\\frac{\\partial" + field + "}{\\partialx}",
    "\\partial_x" + field,
    field + "_x",
    "\\partial" + field + "/\\partialx",
    "\\frac{d" + field + "}{dx}",
    "d" + field + "/dx",
    // Radial variants
    "\\frac{\\partial" + field + "}{\\partialr}",
    "\\partial_r" + field,
    field + "_r",
    "\\partial" + field + "/\\partialr",
    "\\frac{d" + field + "}{dr}",
    "d" + field + "/dr",
  };
}

inline std::vector<std::string> GenerateDYPatterns(const std::string& field) {
  return {
    "\\frac{\\partial" + field + "}{\\partialy}",
    "\\partial_y" + field,
    field + "_y",
    "\\partial" + field + "/\\partialy",
    "\\frac{d" + field + "}{dy}",
    "d" + field + "/dy",
    // Theta variants
    "\\frac{\\partial" + field + "}{\\partialtheta}",
    "\\partial_theta" + field,
    field + "_theta",
    "\\partial" + field + "/\\partialtheta",
    "\\frac{d" + field + "}{dtheta}",
    "d" + field + "/dtheta",
  };
}

inline std::vector<std::string> GenerateDZPatterns(const std::string& field) {
  return {
    "\\frac{\\partial" + field + "}{\\partialz}",
    "\\partial_z" + field,
    field + "_z",
    "\\partial" + field + "/\\partialz",
    "\\frac{d" + field + "}{dz}",
    "d" + field + "/dz",
    // Phi variants
    "\\frac{\\partial" + field + "}{\\partialphi}",
    "\\partial_phi" + field,
    field + "_phi",
    "\\partial" + field + "/\\partialphi",
    "\\frac{d" + field + "}{dphi}",
    "d" + field + "/dphi",
  };
}

inline std::vector<std::string> GenerateDTPatterns(const std::string& field) {
  return {
    "\\frac{\\partial" + field + "}{\\partialt}",
    "\\partial_t" + field,
    field + "_t",
    "\\partial" + field + "/\\partialt",
    "\\frac{d" + field + "}{dt}",
    "d" + field + "/dt",
    "\\dot{" + field + "}",
    "\\dot" + field,
  };
}

inline std::vector<std::string> GenerateD2TPatterns(const std::string& field) {
  return {
    "\\frac{\\partial^2" + field + "}{\\partialt^2}",
    "\\frac{\\partial^2" + field + "}{\\partialt\\partialt}",
    "\\partial_{tt}" + field,
    "\\partial_t\\partial_t" + field,
    field + "_{tt}",
    "\\partial^2" + field + "/\\partialt^2",
    "\\partial^2" + field + "/\\partialt\\partialt",
    "\\frac{d^2" + field + "}{dt^2}",
    "\\frac{d^2" + field + "}{dtdt}",
    "d^2" + field + "/dt^2",
    "d^2" + field + "/dtdt",
    "\\ddot{" + field + "}",
    "\\ddot" + field,
  };
}

inline std::vector<std::string> GenerateDXYPatterns(const std::string& field) {
  return {
    field + "_{xy}",
    field + "_{yx}",
    "\\partial_{xy}" + field,
    "\\partial_{yx}" + field,
    "\\partial_x\\partial_y" + field,
    "\\partial_y\\partial_x" + field,
    "\\frac{\\partial^2" + field + "}{\\partialx\\partialy}",
    "\\frac{\\partial^2" + field + "}{\\partialy\\partialx}",
    "\\partial^2" + field + "/\\partialx\\partialy",
    "\\partial^2" + field + "/\\partialy\\partialx",
    "\\frac{d^2" + field + "}{dxdy}",
    "\\frac{d^2" + field + "}{dydx}",
    "d^2" + field + "/dxdy",
    "d^2" + field + "/dydx",
    // Coordinate variants
    field + "_{rtheta}",
    field + "_{thetar}",
  };
}

inline std::vector<std::string> GenerateDXZPatterns(const std::string& field) {
  return {
    field + "_{xz}",
    field + "_{zx}",
    "\\partial_{xz}" + field,
    "\\partial_{zx}" + field,
    "\\partial_x\\partial_z" + field,
    "\\partial_z\\partial_x" + field,
    "\\frac{\\partial^2" + field + "}{\\partialx\\partialz}",
    "\\frac{\\partial^2" + field + "}{\\partialz\\partialx}",
    "\\partial^2" + field + "/\\partialx\\partialz",
    "\\partial^2" + field + "/\\partialz\\partialx",
    // Coordinate variants
    field + "_{rphi}",
    field + "_{phir}",
  };
}

inline std::vector<std::string> GenerateDYZPatterns(const std::string& field) {
  return {
    field + "_{yz}",
    field + "_{zy}",
    "\\partial_{yz}" + field,
    "\\partial_{zy}" + field,
    "\\partial_y\\partial_z" + field,
    "\\partial_z\\partial_y" + field,
    "\\frac{\\partial^2" + field + "}{\\partialy\\partialz}",
    "\\frac{\\partial^2" + field + "}{\\partialz\\partialy}",
    "\\partial^2" + field + "/\\partialy\\partialz",
    "\\partial^2" + field + "/\\partialz\\partialy",
    // Coordinate variants
    field + "_{thetaphi}",
    field + "_{phitheta}",
  };
}

inline std::vector<std::string> GenerateLaplacianPatterns(const std::string& field) {
  return {
    "\\nabla^2" + field,
    "\\Delta" + field,
    "\\triangle" + field,
  };
}

// Detect field variable from a normalized term
// Returns the field name if found, empty string otherwise
inline std::string DetectFieldVariable(const std::string& normalized_term) {
  // Common single-letter field names to check
  static const std::vector<std::string> single_letter_fields = {
    "u", "v", "w", "p", "q", "c", "s", "a", "b", "f", "g", "h"
  };

  // Check for derivative patterns like field_x, field_xx, field_{xy}
  for (const auto& field : single_letter_fields) {
    // Check for patterns like "v_x", "v_xx", "v_{xy}", etc.
    std::string underscore_pattern = field + "_";
    if (normalized_term.find(underscore_pattern) != std::string::npos) {
      return field;
    }
    // Check for patterns like "\partial_xv", "\partial_{xx}v"
    std::string partial_suffix = "\\partial_x" + field;
    if (normalized_term.find(partial_suffix) != std::string::npos ||
        normalized_term.find("\\partial_y" + field) != std::string::npos ||
        normalized_term.find("\\partial_z" + field) != std::string::npos ||
        normalized_term.find("\\partial_t" + field) != std::string::npos ||
        normalized_term.find("\\partial_r" + field) != std::string::npos) {
      return field;
    }
    // Check for frac patterns
    std::string frac_pattern = "\\partial" + field + "}";
    if (normalized_term.find(frac_pattern) != std::string::npos) {
      return field;
    }
    std::string frac_pattern2 = "d" + field + "}";
    if (normalized_term.find(frac_pattern2) != std::string::npos) {
      return field;
    }
    // Check for standalone field variable (just "v", "w", etc.)
    // This is tricky because we need to ensure it's not part of a larger word
    size_t pos = normalized_term.find(field);
    if (pos != std::string::npos) {
      bool before_ok = (pos == 0) ||
                       !std::isalpha(static_cast<unsigned char>(normalized_term[pos - 1]));
      bool after_ok = (pos + field.length() >= normalized_term.length()) ||
                      normalized_term[pos + field.length()] == '_' ||
                      normalized_term[pos + field.length()] == '^' ||
                      normalized_term[pos + field.length()] == '(' ||
                      normalized_term[pos + field.length()] == '*' ||
                      normalized_term[pos + field.length()] == '+' ||
                      normalized_term[pos + field.length()] == '-' ||
                      normalized_term[pos + field.length()] == ')' ||
                      normalized_term[pos + field.length()] == '}';
      if (before_ok && after_ok) {
        return field;
      }
    }
  }

  return "";
}

}  // namespace LatexPatterns

#endif  // LATEX_PATTERNS_H

