import Image from "next/image";
import NavMock from "../nav-mock";

export async function generateMetadata() {
  const title = "박준우(Park Jun Woo) 한국|Korea";
  const description = "스포트라이트 | 상업 촬영을 위한 전문 플랫폼";
  const metadataBase = new URL("https://yangoos57.github.io");
  const openGraph = {
    title,
    description,
    url: "/spotlite/profile",
    siteName: "spotlite.global",
    images: [
      {
        url: "spotlite/img1_meta.jpeg",
        width: "1054",
        height: "552",
      },
    ],
    type: "website",
  };
  return { metadataBase, title, description, openGraph };
}

export default async function Page() {
  return (
    <>
      <NavMock />
      <div className="w-full">
        <div className="relative max-w-[400px] aspect-[3/4] mx-auto">
          <Image
            src="img1.webp"
            alt="image"
            className="object-cover rounded-xl"
            fill
          />
        </div>
      </div>
    </>
  );
}
