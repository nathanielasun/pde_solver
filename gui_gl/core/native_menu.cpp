#include "native_menu.h"

#ifdef __APPLE__
#import <Cocoa/Cocoa.h>
#endif

// Internal state structure - at file scope for Objective-C access
static struct NativeMenuStateInternal {
  bool installed = false;
  NativeMenuCallbacks callbacks;
#ifdef __APPLE__
  NSMenu* menu = nil;
  id target = nil;
#endif
} g_menu_state;

static NativeMenuStateInternal& MenuState() {
  return g_menu_state;
}

#ifdef __APPLE__
// Objective-C classes must be at global scope
@interface PDEMenuTarget : NSObject <NSMenuItemValidation>
@end

@implementation PDEMenuTarget
- (void)onMenuAction:(id)sender {
  NSMenuItem* item = (NSMenuItem*)sender;
  auto& state = MenuState();
  if (state.callbacks.on_action) {
    state.callbacks.on_action(static_cast<AppAction>(item.tag));
  }
}

- (BOOL)validateMenuItem:(NSMenuItem*)item {
  if (item.action != @selector(onMenuAction:)) {
    return YES;
  }
  auto& state = MenuState();
  AppAction action = static_cast<AppAction>(item.tag);
  const bool enabled = state.callbacks.can_execute ? state.callbacks.can_execute(action) : true;
  if (state.callbacks.is_checked) {
    const bool checked = state.callbacks.is_checked(action);
    [item setState:checked ? NSControlStateValueOn : NSControlStateValueOff];
  } else {
    [item setState:NSControlStateValueOff];
  }
  return enabled;
}
@end

namespace {

NSString* AppName() {
  NSString* name = [[NSBundle mainBundle] objectForInfoDictionaryKey:@"CFBundleName"];
  if (!name || [name length] == 0) {
    name = @"PDE Solver";
  }
  return name;
}

NSMenuItem* AddMenuItem(NSMenu* menu, NSString* title, AppAction action,
                        NSString* key, NSEventModifierFlags mods, id target) {
  NSString* key_equiv = key ? key : @"";
  NSMenuItem* item = [[NSMenuItem alloc] initWithTitle:title
                                                action:@selector(onMenuAction:)
                                         keyEquivalent:key_equiv];
  [item setTarget:target];
  [item setTag:static_cast<NSInteger>(action)];
  if ([key_equiv length] > 0) {
    [item setKeyEquivalentModifierMask:mods];
  } else {
    [item setKeyEquivalentModifierMask:0];
  }
  [menu addItem:item];
  return item;
}

}  // namespace
#endif

NativeMenu& NativeMenu::Instance() {
  static NativeMenu instance;
  return instance;
}

bool NativeMenu::IsInstalled() const {
  return MenuState().installed;
}

void NativeMenu::Install(GLFWwindow* window, const NativeMenuCallbacks& callbacks) {
  (void)window;
#ifdef __APPLE__
  auto& state = MenuState();
  if (state.installed) {
    state.callbacks = callbacks;
    return;
  }
  state.callbacks = callbacks;

  @autoreleasepool {
    if (!NSApp) {
      [NSApplication sharedApplication];
    }
    PDEMenuTarget* target = [[PDEMenuTarget alloc] init];
    state.target = target;

    NSMenu* main_menu = [[NSMenu alloc] initWithTitle:@""];
    state.menu = main_menu;
    [NSApp setMainMenu:main_menu];

    NSString* app_name = AppName();
    NSMenuItem* app_item = [[NSMenuItem alloc] initWithTitle:@"" action:nil keyEquivalent:@""];
    [main_menu addItem:app_item];
    NSMenu* app_menu = [[NSMenu alloc] initWithTitle:app_name];
    [app_item setSubmenu:app_menu];

    AddMenuItem(app_menu, [NSString stringWithFormat:@"About %@", app_name],
                AppAction::kAbout, @"", 0, target);
    [app_menu addItem:[NSMenuItem separatorItem]];
    AddMenuItem(app_menu, @"Preferences…", AppAction::kGoPreferencesTab,
                @",", NSEventModifierFlagCommand, target);
    [app_menu addItem:[NSMenuItem separatorItem]];
    AddMenuItem(app_menu, [NSString stringWithFormat:@"Quit %@", app_name],
                AppAction::kQuit, @"q", NSEventModifierFlagCommand, target);

    // File menu
    NSMenuItem* file_item = [[NSMenuItem alloc] initWithTitle:@"File" action:nil keyEquivalent:@""];
    [main_menu addItem:file_item];
    NSMenu* file_menu = [[NSMenu alloc] initWithTitle:@"File"];
    [file_item setSubmenu:file_menu];
    AddMenuItem(file_menu, @"New Session", AppAction::kNewSession,
                @"n", NSEventModifierFlagCommand, target);
    AddMenuItem(file_menu, @"Load Latest Result", AppAction::kLoadLatest,
                @"l", NSEventModifierFlagCommand, target);
    AddMenuItem(file_menu, @"Export Image", AppAction::kExportImage, @"", 0, target);
    [file_menu addItem:[NSMenuItem separatorItem]];
    AddMenuItem(file_menu, @"Load Run Config", AppAction::kLoadRunConfig, @"", 0, target);
    AddMenuItem(file_menu, @"Save Run Config", AppAction::kSaveRunConfig, @"", 0, target);
    [file_menu addItem:[NSMenuItem separatorItem]];
    AddMenuItem(file_menu, @"Load UI Config", AppAction::kLoadUIConfig,
                @"o", NSEventModifierFlagCommand, target);
    AddMenuItem(file_menu, @"Save UI Config", AppAction::kSaveUIConfig,
                @"s", NSEventModifierFlagCommand | NSEventModifierFlagShift, target);

    // Edit menu
    NSMenuItem* edit_item = [[NSMenuItem alloc] initWithTitle:@"Edit" action:nil keyEquivalent:@""];
    [main_menu addItem:edit_item];
    NSMenu* edit_menu = [[NSMenu alloc] initWithTitle:@"Edit"];
    [edit_item setSubmenu:edit_menu];
    AddMenuItem(edit_menu, @"Undo", AppAction::kUndo,
                @"z", NSEventModifierFlagCommand, target);
    AddMenuItem(edit_menu, @"Redo", AppAction::kRedo,
                @"z", NSEventModifierFlagCommand | NSEventModifierFlagShift, target);
    [edit_menu addItem:[NSMenuItem separatorItem]];
    AddMenuItem(edit_menu, @"Copy PDE", AppAction::kCopyPDE, @"", 0, target);
    AddMenuItem(edit_menu, @"Paste PDE", AppAction::kPastePDE, @"", 0, target);

    // View menu
    NSMenuItem* view_item = [[NSMenuItem alloc] initWithTitle:@"View" action:nil keyEquivalent:@""];
    [main_menu addItem:view_item];
    NSMenu* view_menu = [[NSMenu alloc] initWithTitle:@"View"];
    [view_item setSubmenu:view_menu];
    AddMenuItem(view_menu, @"Reset View", AppAction::kResetView,
                @"r", NSEventModifierFlagCommand, target);
    AddMenuItem(view_menu, @"Orthographic Projection", AppAction::kToggleOrtho, @"", 0, target);
    AddMenuItem(view_menu, @"Show Domain Grid", AppAction::kToggleGrid, @"", 0, target);
    AddMenuItem(view_menu, @"Lock Z Domain", AppAction::kToggleZLock, @"", 0, target);

    // Go menu
    NSMenuItem* go_item = [[NSMenuItem alloc] initWithTitle:@"Go" action:nil keyEquivalent:@""];
    [main_menu addItem:go_item];
    NSMenu* go_menu = [[NSMenu alloc] initWithTitle:@"Go"];
    [go_item setSubmenu:go_menu];
    AddMenuItem(go_menu, @"Main Tab", AppAction::kGoMainTab,
                @"1", NSEventModifierFlagCommand, target);
    AddMenuItem(go_menu, @"Inspect Tab", AppAction::kGoInspectTab,
                @"2", NSEventModifierFlagCommand, target);
    AddMenuItem(go_menu, @"Preferences Tab", AppAction::kGoPreferencesTab,
                @"3", NSEventModifierFlagCommand, target);

    // Tools menu
    NSMenuItem* tools_item = [[NSMenuItem alloc] initWithTitle:@"Tools" action:nil keyEquivalent:@""];
    [main_menu addItem:tools_item];
    NSMenu* tools_menu = [[NSMenu alloc] initWithTitle:@"Tools"];
    [tools_item setSubmenu:tools_menu];
    AddMenuItem(tools_menu, @"Solve PDE", AppAction::kSolve,
                @"s", NSEventModifierFlagCommand, target);
    AddMenuItem(tools_menu, @"Stop Solver", AppAction::kStop, @"", 0, target);
    [tools_menu addItem:[NSMenuItem separatorItem]];
    AddMenuItem(tools_menu, @"Benchmarks", AppAction::kOpenBenchmarks, @"", 0, target);
    AddMenuItem(tools_menu, @"Comparison Tools", AppAction::kOpenComparisonTools, @"", 0, target);
    AddMenuItem(tools_menu, @"Advanced Inspection", AppAction::kOpenAdvancedInspection, @"", 0, target);
    AddMenuItem(tools_menu, @"UI Configuration", AppAction::kOpenUIConfigPanel, @"", 0, target);
    AddMenuItem(tools_menu, @"Validate UI Config", AppAction::kValidateUIConfig, @"", 0, target);

    // Window menu
    NSMenuItem* window_item = [[NSMenuItem alloc] initWithTitle:@"Window" action:nil keyEquivalent:@""];
    [main_menu addItem:window_item];
    NSMenu* window_menu = [[NSMenu alloc] initWithTitle:@"Window"];
    [window_item setSubmenu:window_menu];
    [NSApp setWindowsMenu:window_menu];
    AddMenuItem(window_menu, @"Show Left Panel", AppAction::kToggleLeftPanel, @"", 0, target);
    AddMenuItem(window_menu, @"Reset Layout", AppAction::kResetLayout, @"", 0, target);
    [window_menu addItem:[NSMenuItem separatorItem]];
    AddMenuItem(window_menu, @"Enable Docking UI", AppAction::kToggleDockingUI, @"", 0, target);
    NSMenuItem* layout_item = [[NSMenuItem alloc] initWithTitle:@"Layout Presets"
                                                         action:nil
                                                  keyEquivalent:@""];
    [window_menu addItem:layout_item];
    NSMenu* layout_menu = [[NSMenu alloc] initWithTitle:@"Layout Presets"];
    [layout_item setSubmenu:layout_menu];
    AddMenuItem(layout_menu, @"Default Layout", AppAction::kLayoutDefault, @"", 0, target);
    AddMenuItem(layout_menu, @"Inspection Layout", AppAction::kLayoutInspection, @"", 0, target);
    AddMenuItem(layout_menu, @"Dual Viewer", AppAction::kLayoutDualViewer, @"", 0, target);
    AddMenuItem(layout_menu, @"Minimal", AppAction::kLayoutMinimal, @"", 0, target);
    AddMenuItem(layout_menu, @"Full Configuration", AppAction::kLayoutFull, @"", 0, target);

    // Help menu
    NSMenuItem* help_item = [[NSMenuItem alloc] initWithTitle:@"Help" action:nil keyEquivalent:@""];
    [main_menu addItem:help_item];
    NSMenu* help_menu = [[NSMenu alloc] initWithTitle:@"Help"];
    [help_item setSubmenu:help_menu];
    [NSApp setHelpMenu:help_menu];
    AddMenuItem(help_menu, @"Help Search", AppAction::kHelpSearch, @"", 0, target);
  }

  state.installed = true;
#else
  (void)callbacks;
#endif
}
