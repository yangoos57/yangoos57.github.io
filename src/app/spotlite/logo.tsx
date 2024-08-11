import Image from "next/image";
import logoImage from "../../../public/spotlite/logo.png";
import Link from "next/link";
export default function Logo() {
  return (
    <Link href="/spotlite" className="relative h-[40px] aspect-[4.25/1]">
      <Image
        src={logoImage}
        alt="logo"
        className="object-contain"
        fill
        unoptimized
      />
    </Link>
  );
}
