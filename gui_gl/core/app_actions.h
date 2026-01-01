#ifndef APP_ACTIONS_H
#define APP_ACTIONS_H

enum class AppAction : int {
  kSolve,
  kStop,
  kLoadLatest,
  kResetView,
  kUndo,
  kRedo,
  kHelpSearch,
  kNewSession,
  kLoadUIConfig,
  kSaveUIConfig,
  kLoadRunConfig,
  kSaveRunConfig,
  kExportImage,
  kQuit,
  kCopyPDE,
  kPastePDE,
  kToggleOrtho,
  kToggleGrid,
  kToggleZLock,
  kToggleLeftPanel,
  kGoMainTab,
  kGoInspectTab,
  kGoPreferencesTab,
  kOpenBenchmarks,
  kOpenComparisonTools,
  kOpenAdvancedInspection,
  kOpenUIConfigPanel,
  kValidateUIConfig,
  kToggleDockingUI,
  kLayoutDefault,
  kLayoutInspection,
  kLayoutDualViewer,
  kLayoutMinimal,
  kLayoutFull,
  kResetLayout,
  kAbout
};

#endif  // APP_ACTIONS_H
