#ifndef UI_STYLE_H
#define UI_STYLE_H

#include "imgui.h"
#include <string>
#include <functional>

// ============================================================================
// SPACING SYSTEM (4px/8px grid)
// ============================================================================

namespace UISpacing {
  // Base spacing unit (4px grid)
  constexpr float Unit = 4.0f;
  
  // Spacing scale
  constexpr float XS = 4.0f;   // Tight spacing (related items)
  constexpr float SM = 8.0f;   // Standard spacing (form fields)
  constexpr float MD = 12.0f;  // Medium spacing
  constexpr float LG = 16.0f;  // Section spacing (between panels)
  constexpr float XL = 24.0f;  // Major section spacing
  constexpr float XXL = 32.0f; // Panel spacing (between top-level panels)
  
  // Padding
  constexpr float InputPaddingY = 8.0f;
  constexpr float InputPaddingX = 12.0f;
  constexpr float ButtonPaddingY = 10.0f;
  constexpr float ButtonPaddingX = 20.0f;
  constexpr float PanelPadding = 16.0f;
  constexpr float WindowPadding = 20.0f;
}

// ============================================================================
// TYPOGRAPHY SYSTEM
// ============================================================================

namespace UITypography {
  // Font sizes (in pixels)
  constexpr float Small = 11.0f;      // Help text, tooltips, disabled text
  constexpr float Regular = 13.0f;    // Body text, inputs, buttons
  constexpr float Medium = 15.0f;     // Section headers, labels
  constexpr float Large = 18.0f;      // Panel titles, major headings
  constexpr float ExtraLarge = 24.0f; // Window title, splash screens
  
  // Line height multipliers
  constexpr float LineHeightTight = 1.4f;
  constexpr float LineHeightNormal = 1.5f;
  constexpr float LineHeightRelaxed = 1.6f;
  
  // Letter spacing (for uppercase labels)
  constexpr float LetterSpacingNormal = 0.0f;
  constexpr float LetterSpacingWide = 0.5f;
  constexpr float LetterSpacingWider = 1.0f;
}

// ============================================================================
// BUTTON STYLING
// ============================================================================

namespace UIButton {
  // Button sizes
  enum class Size {
    Small,   // 28-32px height
    Medium,  // 36-40px height (default)
    Large    // 44-48px height
  };
  
  // Button variants
  enum class Variant {
    Primary,    // Prominent, accent color
    Secondary,  // Outlined or subtle background
    Tertiary,   // Text-only
    Danger      // Red variant for destructive actions
  };
  
  // Apply button styling
  void PushStyle(Size size = Size::Medium, Variant variant = Variant::Primary);
  void PopStyle();
  
  // Helper to create styled button
  bool Button(const char* label, Size size = Size::Medium, Variant variant = Variant::Primary);
  bool Button(const char* label, const ImVec2& size, Size button_size = Size::Medium, Variant variant = Variant::Primary);
}

// ============================================================================
// INPUT FIELD STYLING
// ============================================================================

namespace UIInput {
  // Input validation states
  enum class ValidationState {
    None,     // No validation
    Valid,    // Valid input
    Warning,  // Warning state
    Error     // Error state
  };
  
  // Apply input styling
  void PushStyle(ValidationState state = ValidationState::None);
  void PopStyle();
  
  // Helper for styled input
  bool InputText(const char* label, std::string* str, ImGuiInputTextFlags flags = 0);
  bool InputTextMultiline(const char* label, std::string* str, const ImVec2& size, ImGuiInputTextFlags flags = 0);
  bool InputFloat(const char* label, float* v, float step = 0.0f, float step_fast = 0.0f, const char* format = "%.3f");
  bool InputDouble(const char* label, double* v, double step = 0.0, double step_fast = 0.0, const char* format = "%.6f");
  bool InputInt(const char* label, int* v, int step = 1, int step_fast = 10);
  bool InputInt2(const char* label, int v[2]);
  bool SliderFloat(const char* label, float* v, float v_min, float v_max, const char* format = "%.3f");
  bool SliderInt(const char* label, int* v, int v_min, int v_max);
  bool Combo(const char* label, int* current_item, const char* const items[], int items_count);
}

// ============================================================================
// TOAST NOTIFICATION SYSTEM
// ============================================================================

namespace UIToast {
  // Toast types
  enum class Type {
    Success,
    Warning,
    Error,
    Info
  };
  
  // Toast message structure
  struct Toast {
    std::string message;
    Type type;
    float duration;  // Duration in seconds (0 = auto)
    float elapsed;  // Time elapsed
    int id;         // Unique ID
  };
  
  // Initialize toast system (call once at startup)
  void Initialize();
  
  // Show a toast notification
  void Show(Type type, const std::string& message, float duration = 3.0f);
  
  // Render toasts (call in main render loop)
  void Render();
  
  // Clear all toasts
  void Clear();
}

// ============================================================================
// ICON SYSTEM (Phase 2)
// ============================================================================

namespace UIIcon {
  // Icon types (using Unicode symbols for now, can be replaced with icon font)
  namespace Symbol {
    constexpr const char* Play = "▶";
    constexpr const char* Pause = "⏸";
    constexpr const char* Stop = "⏹";
    constexpr const char* Load = "📁";
    constexpr const char* Save = "💾";
    constexpr const char* Settings = "⚙";
    constexpr const char* Info = "ℹ";
    constexpr const char* Warning = "⚠";
    constexpr const char* Error = "✗";
    constexpr const char* Success = "✓";
    constexpr const char* Close = "✕";
    constexpr const char* ChevronRight = "▶";
    constexpr const char* ChevronDown = "▼";
    constexpr const char* Plus = "+";
    constexpr const char* Minus = "−";
    constexpr const char* Refresh = "↻";
    constexpr const char* Search = "🔍";
    constexpr const char* Edit = "✎";
    constexpr const char* Delete = "🗑";
  }
  
  // Icon sizes
  enum class Size {
    Small = 16,   // Inline with text
    Medium = 20,  // Buttons and controls
    Large = 24,   // Section headers
    ExtraLarge = 32  // Major actions
  };
  
  // Draw icon with text
  void Text(const char* icon, const char* text, Size size = Size::Medium);
  
  // Draw icon only
  void Icon(const char* icon, Size size = Size::Medium, const ImVec4& color = ImVec4(1, 1, 1, 1));
  
  // Draw icon button
  bool Button(const char* icon, const char* label = nullptr, Size size = Size::Medium);
}

// ============================================================================
// PANEL STYLING (Phase 2)
// ============================================================================

namespace UIPanel {
  // Panel elevation levels
  enum class Elevation {
    Flat = 0,      // No shadow
    Subtle = 1,    // Subtle shadow (hover states)
    Medium = 2,    // Medium shadow (active/dragging)
    Strong = 3    // Strong shadow (modals, popups)
  };
  
  // Begin a styled panel
  bool Begin(const char* name, bool* p_open = nullptr, ImGuiWindowFlags flags = 0, Elevation elevation = Elevation::Flat);
  void End();
  
  // Begin a styled child panel
  bool BeginChild(const char* str_id, const ImVec2& size = ImVec2(0, 0), bool border = false, ImGuiWindowFlags flags = 0, Elevation elevation = Elevation::Flat);
  void EndChild();
  
  // Draw panel with card styling
  void BeginCard(const char* label = nullptr, bool border = true, Elevation elevation = Elevation::Subtle);
  void EndCard();
  
  // Draw shadow (helper function)
  void DrawShadow(const ImVec2& min, const ImVec2& max, Elevation elevation, float rounding = 8.0f);
}

// ============================================================================
// ANIMATION SYSTEM (Phase 2)
// ============================================================================

namespace UIAnimation {
  // Initialize animation system
  void Initialize();
  
  // Easing functions
  float EaseInOut(float t);
  float EaseOut(float t);
  float EaseIn(float t);
  
  // Animated value (0.0 to 1.0)
  float Animate(const char* id, float target, float speed = 5.0f);
  
  // Animated color
  ImVec4 AnimateColor(const char* id, const ImVec4& target, float speed = 5.0f);
  
  // Animated float
  float AnimateFloat(const char* id, float target, float speed = 5.0f);
  
  // Reset animation
  void Reset(const char* id);
  
  // Update all animations (call once per frame)
  void Update(float delta_time);
}

// ============================================================================
// EMPTY STATES (Phase 2)
// ============================================================================

namespace UIEmptyState {
  // Empty state types
  enum class Type {
    NoData,        // No data available
    NoResults,     // No search results
    NoLogs,        // Empty log
    Loading,       // Loading state
    Error,         // Error state
    Empty          // Generic empty
  };
  
  // Show empty state
  void Show(Type type, const char* title = nullptr, const char* message = nullptr, const char* action_label = nullptr, std::function<void()> action = nullptr);
  
  // Show loading skeleton
  void ShowSkeleton(int lines = 3);
}

// ============================================================================
// STATUS BADGES (Phase 2)
// ============================================================================

namespace UIBadge {
  // Badge types
  enum class Type {
    Success,
    Warning,
    Error,
    Info,
    Neutral
  };
  
  // Draw status badge
  void Draw(const char* text, Type type = Type::Neutral, bool small = false);
  
  // Draw pill badge (for tags)
  void DrawPill(const char* text, Type type = Type::Neutral);
}

// ============================================================================
// STYLE SYSTEM INITIALIZATION
// ============================================================================

namespace UIStyle {
  // Initialize the style system (call once after ImGui::CreateContext())
  void Initialize();
  
  // Apply professional styling to ImGui style
  void ApplyProfessionalStyle();
  
  // Apply spacing to current style
  void ApplySpacing();
  
  // Apply typography to current style
  void ApplyTypography();
  
  // Apply panel styling (Phase 2)
  void ApplyPanelStyling();
}

// ============================================================================
// HELP SYSTEM (Phase 3)
// ============================================================================

namespace UIHelp {
  // Help entry structure
  struct HelpEntry {
    std::string id;
    std::string title;
    std::string content;
    std::string category;
    std::vector<std::string> keywords;
    std::string shortcut;  // Keyboard shortcut if applicable
  };
  
  // Initialize help system
  void Initialize();

  // Open the help search UI (e.g., from an F1 handler).
  void OpenSearch();
  
  // Register help entry
  void RegisterHelp(const HelpEntry& entry);
  
  // Show tooltip with help content
  void ShowTooltip(const char* help_id);
  
  // Show help icon button (question mark)
  bool HelpIcon(const char* help_id, const char* tooltip_text = nullptr);
  
  // Show contextual help panel
  void ShowHelpPanel(const char* help_id);
  
  // Show help search dialog
  void ShowHelpSearch();
  
  // Get help entry by ID
  const HelpEntry* GetHelp(const char* help_id);
}

// ============================================================================
// ACCESSIBILITY (Phase 3)
// ============================================================================

namespace UIAccessibility {
  // Accessibility settings
  struct AccessibilitySettings {
    bool high_contrast = false;
    bool reduced_motion = false;
    float text_scale = 1.0f;
    bool screen_reader_mode = false;
    bool large_click_targets = false;
  };
  
  // Get current accessibility settings
  AccessibilitySettings& GetSettings();
  
  // Apply accessibility settings to style
  void ApplySettings();
  
  // Check if reduced motion is enabled
  bool IsReducedMotion();
  
  // Get text scale factor
  float GetTextScale();
  
  // Check if large click targets enabled
  bool UseLargeClickTargets();
}

// ============================================================================
// ADVANCED ANIMATIONS (Phase 3)
// ============================================================================

namespace UIAdvancedAnimation {
  // Animation types
  enum class Type {
    Fade,
    Slide,
    Scale,
    Rotate,
    Bounce
  };
  
  // Animate panel expand/collapse
  float AnimatePanelHeight(const char* id, float target_height, float speed = 5.0f);
  
  // Animate fade in/out
  float AnimateFade(const char* id, float target_alpha, float speed = 5.0f);
  
  // Animate slide
  ImVec2 AnimateSlide(const char* id, const ImVec2& target_pos, float speed = 5.0f);
  
  // Animate scale
  float AnimateScale(const char* id, float target_scale, float speed = 5.0f);
  
  // Animate rotation
  float AnimateRotation(const char* id, float target_angle, float speed = 5.0f);
  
  // Pulse animation (for loading indicators)
  float AnimatePulse(const char* id, float min = 0.5f, float max = 1.0f, float speed = 2.0f);
  
  // Shake animation (for errors)
  ImVec2 AnimateShake(const char* id, float intensity = 5.0f, float speed = 10.0f);
}

// ============================================================================
// CUSTOM COMPONENTS (Phase 3)
// ============================================================================

namespace UICustomComponents {
  // Searchable dropdown
  bool SearchableCombo(const char* label, int* current_item, const char* const items[], int items_count, const char* hint = "Search...");
  
  // Number input with increment/decrement buttons
  bool NumberInput(const char* label, double* value, double step = 1.0, double step_fast = 10.0, const char* format = "%.6f");
  
  // Toggle switch (replaces checkbox for better UX)
  bool ToggleSwitch(const char* label, bool* value);
  
  // Color picker button (opens color picker popup)
  bool ColorButton(const char* label, ImVec4* color, ImGuiColorEditFlags flags = 0);
  
  // Progress bar with percentage
  void ProgressBarWithText(float fraction, const ImVec2& size = ImVec2(-1, 0), const char* overlay = nullptr);
  
  // Loading spinner
  void LoadingSpinner(const char* label = nullptr, float radius = 10.0f, float thickness = 2.0f);
  
  // Collapsible section with icon
  bool CollapsingHeaderWithIcon(const char* label, const char* icon, ImGuiTreeNodeFlags flags = 0);
  
  // Tab bar with icons
  bool BeginTabBarWithIcons(const char* str_id, ImGuiTabBarFlags flags = 0);
  bool TabItemWithIcon(const char* label, const char* icon, bool* p_open = nullptr, ImGuiTabItemFlags flags = 0);
}

#endif // UI_STYLE_H
