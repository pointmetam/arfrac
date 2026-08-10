import SwiftUI

struct ARPreviewContainer: View {
    @ObservedObject var state: MakeupState

    var body: some View {
        Group {
            if state.usesLip2DCaptureView {
                ARFaceMakeupView(state: state)
            } else {
                ARKitFaceMakeupView(state: state)
            }
        }
    }
}

#Preview {
    ARPreviewContainer(state: MakeupState())
}
