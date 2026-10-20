import SwiftUI
import UIKit

@main
struct HandPilotApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
                // Nothing touches the phone during a session -- with the rear
                // camera its screen even faces away. Without this it sleeps
                // mid-use and the camera stops.
                .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
                .onDisappear { UIApplication.shared.isIdleTimerDisabled = false }
        }
    }
}
