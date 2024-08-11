import Image from "next/image";

export default function Logo() {
  return (
    <div className="relative h-[40px] aspect-[4.25/1] ">
      <Image
        src={"spotlite/logo.png"}
        alt="logo"
        className="object-contain"
        fill
        unoptimized
      />
    </div>
  );
}
