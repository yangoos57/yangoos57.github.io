import Link from "next/link";
import CoverImage from "./cover-image";
import DateFormatter from "./date-formatter";

type Props = {
  title: string;
  thumbnail: string;
  date: string;
  desc: string;
  category: string[];
  slug: string;
};

function Info({ category, date }: { category: string[]; date: string }) {
  return (
    <div className="text-sm md:text-base flex gap-x-2 text-gray-500  pb-2">
      <div className="text-nav font-medium">
        {category.map((c) => (
          <span className="capitalize me-2" key={c}>
            {c}
          </span>
        ))}
      </div>
      <DateFormatter dateString={date} />
    </div>
  );
}

export function PostPreview({
  title,
  thumbnail,
  date,
  desc,
  category,
  slug,
}: Props) {
  return (
    <>
      <Link href={`/blog/${slug}`} className="group py-4 px-10">
        <Info category={category} date={date} />
        <h1 className="text-lg md:text-xl font-bold group-hover:underline mb-3 text-black/90 line-clamp-1">
          {title}
        </h1>
      </Link>
    </>
  );
}
