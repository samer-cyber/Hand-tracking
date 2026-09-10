import SwiftUI
import UIKit

@main
struct HandPilotApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                // The phone is pointed at you and its screen is facing away, so
                // nothing will ever touch it during a session. Without this it
                // sleeps mid-use and the camera stops.
                .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
                .onDisappear { UIApplication.shared.isIdleTimerDisabled = false }
        }
    }
}
