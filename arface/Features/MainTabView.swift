import SwiftUI

struct MainTabView: View {
    var body: some View {
        TabView {
            NavigationStack {
                HomeView()
            }
            .tabItem {
                Label("首页", systemImage: "house")
            }

            NavigationStack {
                CreatePostView()
            }
            .tabItem {
                Label("发布", systemImage: "plus.circle")
            }

            NavigationStack {
                ProfileView()
            }
            .tabItem {
                Label("我的", systemImage: "person")
            }
        }
        .tint(BrandColor.primary)
    }
}

#Preview {
    MainTabView()
}
