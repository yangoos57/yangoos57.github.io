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

export function PostPreview({ title, thumbnail, date, desc, category, slug }: Props) {
    return (
        <>
            <Link as={`/blog/${slug}`} href="/blog/[slug]">
                <div className="group bg-white rounded-md aspect-[4/5] w-full flex flex-col">
                    <div className="basis-[50%] overflow-hidden relative ">
                        <div className="absolute inset-0 rounded-md group-hover:bg-black/25 z-10 transition-all duration-500"></div>
                        <CoverImage slug={slug} title={title} src={thumbnail} />
                    </div>
                    <div className="basis-[45%] grow mx-5 overflow-hidden relative ">
                        <div className="text-gray-500 capitalize py-1">
                            {category.map((c) => (
                                <span className="capitalize me-2" key={c}>
                                    {c}
                                </span>
                            ))}
                        </div>
                        <h3 className="text-2xl font-bold group-hover:underline mb-3 ">{title}</h3>
                        <p className="leading-relaxed mb-4 ">{desc}</p>
                        <div className="w-full bg-white absolute bottom-0 text-gray-500 py-2 ">
                            <DateFormatter dateString={date} />
                        </div>
                    </div>
                </div>
            </Link>
        </>
    );
}
