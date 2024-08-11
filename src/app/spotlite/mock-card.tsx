import Link from "next/link";
import EmblaCarousel, { CarouselImage } from "@/components/carousel/carousel";

export function MockCard({ image }: { image: string }) {
  return (
    <Link
      href="spotlite/profile"
      className="flex flex-col min-w-full sm:min-w-[300px] w-full aspect-[3/4]"
    >
      <CarouselImage
        alt="photo"
        src={`spotlite/${image}`}
        className="object-fill pe-2"
        fill
        unoptimized
      />
      <div className="mx-auto pb-2">Model</div>
    </Link>
  );
}

export function Cards() {
  const Arr = [
    "img1.webp",
    "img2.webp",
    "img1.webp",
    "img2.webp",
    "img1.webp",
    "img2.webp",
    "img1.webp",
    "img2.webp",
  ];
  return Arr.map((image, idx) => <MockCard key={idx} image={image} />);
}

export default function MockCardBox({
  children,
}: {
  children: React.ReactNode;
}) {
  return <EmblaCarousel type="single">{children}</EmblaCarousel>;
}
