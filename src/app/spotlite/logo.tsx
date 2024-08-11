import Image from "next/image";

import Link from "next/link";
export default function Logo() {
  return (
    <Link href="/spotlite" className="relative h-[40px] aspect-[4.25/1]">
      <Image
        src="/spotlite/logo.png"
        alt="logo"
        className="object-contain"
        fill
        unoptimized
      />
    </Link>
  );
}
