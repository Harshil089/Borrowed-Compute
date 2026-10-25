//! Lets the app decide whether a quit really quits.
//!
//! macOS asks the app delegate `applicationShouldTerminate:` before quitting from ⌘Q, the
//! Dock's Quit, or another app's quit request. Tauri's delegate does not implement it, so
//! these quits never reach `RunEvent::ExitRequested` and cannot be prevented there. This adds
//! the method to the delegate's class at startup.
//!
//! Quits for logout, restart and shutdown always go through, so the app never blocks them.

use std::sync::OnceLock;

use objc2::runtime::{AnyObject, Imp, Sel};
use objc2::{class, msg_send, sel};

/// Returns true to let the app quit.
type Decide = Box<dyn Fn() -> bool + Send + Sync>;

static DECIDE: OnceLock<Decide> = OnceLock::new();

const NS_TERMINATE_CANCEL: usize = 0;
const NS_TERMINATE_NOW: usize = 1;
/// `kAEQuitReason`: present on the quit event when the system is logging out or shutting down.
const QUIT_REASON: u32 = u32::from_be_bytes(*b"why?");

/// Must run on the main thread after the event loop has set the app delegate, as in Tauri's
/// `setup`.
pub fn install(decide: impl Fn() -> bool + Send + Sync + 'static) {
    if DECIDE.set(Box::new(decide)).is_err() {
        return;
    }
    // SAFETY: plain Objective-C runtime calls on the main thread. The added method has the
    // signature AppKit expects: (id self, SEL _cmd, NSApplication *sender) -> NSUInteger.
    unsafe {
        let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];
        let delegate: *mut AnyObject = msg_send![app, delegate];
        let Some(delegate) = delegate.as_ref() else { return };
        let cls = delegate.class() as *const _ as *mut objc2::runtime::AnyClass;
        let imp: Imp = std::mem::transmute(
            should_terminate as unsafe extern "C-unwind" fn(*mut AnyObject, Sel, *mut AnyObject) -> usize,
        );
        objc2::ffi::class_addMethod(cls, sel!(applicationShouldTerminate:), imp, c"Q@:@".as_ptr());
    }
}

unsafe extern "C-unwind" fn should_terminate(_this: *mut AnyObject, _cmd: Sel, _sender: *mut AnyObject) -> usize {
    if system_is_ending_session() {
        return NS_TERMINATE_NOW;
    }
    match DECIDE.get() {
        Some(decide) if !decide() => NS_TERMINATE_CANCEL,
        _ => NS_TERMINATE_NOW,
    }
}

fn system_is_ending_session() -> bool {
    // SAFETY: reads the Apple event being handled, if any; every pointer is checked.
    unsafe {
        let manager: *mut AnyObject = msg_send![class!(NSAppleEventManager), sharedAppleEventManager];
        let event: *mut AnyObject = msg_send![manager, currentAppleEvent];
        if event.is_null() {
            return false;
        }
        let reason: *mut AnyObject = msg_send![event, attributeDescriptorForKeyword: QUIT_REASON];
        !reason.is_null()
    }
}
