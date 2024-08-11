import NavMock from "./nav-mock";
import MockCardBox, { Cards } from "./mock-card";
import MockCatsBox, { MockCats } from "./mock-cat";

export async function generateMetadata() {
  const title = "박준우(Park Jun Woo) 한국|Korea";
  const description = "스포트라이트 | 상업 촬영을 위한 전문 플랫폼";
  const metadataBase = new URL("https://yangoos57.github.io");
  const openGraph = {
    title,
    description,
    url: "/spotlite",
    siteName: "spotlite.global",
    type: "website",
  };
  return { metadataBase, title, description, openGraph };
}

export default function Page() {
  return (
    <>
      <NavMock />
      <div className="w-full h-full bg-gray-100">
        <MockCatsBox>
          <MockCats />
        </MockCatsBox>
        <div className="pb-1 font-semibold text-lg pt-2">
          새로운 모델을 만나보세요.
        </div>
        <MockCardBox>
          <Cards />
        </MockCardBox>
      </div>
    </>
  );
}
