import Foundation

struct FeaturedLook: Identifiable, Hashable {
    let id: String
    let title: String
    let subtitle: String
    let creatorName: String
    let tutorialURL: URL
    let heroImageName: String
    let tags: [String]
    let specs: [LookSpec]
    let products: [LookProduct]
}

struct LookSpec: Identifiable, Hashable {
    let id = UUID()
    let title: String
    let value: String
}

struct LookProduct: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let category: String
    let price: String
}

enum MockData {
    static let featuredLook = FeaturedLook(
        id: "midnight-rose",
        title: "午夜玫瑰幻影",
        subtitle: "冷调酒红唇搭配微光眼妆，适合夜晚聚会和拍照出片。",
        creatorName: "@LilyMuse",
        tutorialURL: URL(string: "https://www.xiaohongshu.com")!,
        heroImageName: "MidnightRoseHero",
        tags: ["法式复古红唇", "丝缎底妆", "晚宴妆"],
        specs: [
            LookSpec(title: "底妆", value: "Matte"),
            LookSpec(title: "唇妆", value: "Deep Plum"),
            LookSpec(title: "眼妆", value: "Shimmer"),
            LookSpec(title: "风格", value: "Editorial")
        ],
        products: [
            LookProduct(name: "兰蔻菁纯丝绒唇膏", category: "口红 #274", price: "¥380"),
            LookProduct(name: "雾感塑形眼影盘", category: "眼妆", price: "¥260"),
            LookProduct(name: "柔焦持妆粉底", category: "底妆", price: "¥320")
        ]
    )
}
