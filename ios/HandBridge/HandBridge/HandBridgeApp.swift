import SwiftUI
import UIKit

@main
struct HandBridgeApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                // Hand tracking is useless if the screen sleeps mid-session.
                .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
                .onDisappear { UIApplication.shared.isIdleTimerDisabled = false }
        }
    }
}
