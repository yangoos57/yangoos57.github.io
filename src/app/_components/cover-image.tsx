import cn from "classnames";
import Link from "next/link";
import Image from "next/image";

type Props = {
    title: string;
    src: string;
    slug?: string;
};

const CoverImage = ({ title, src, slug }: Props) => {
    return (
        <div className="sm:mx-0">
            <Image
                src={src}
                alt={`Cover Image for ${title}`}
                className={cn("w-full", "group-hover:scale-[105%]", {
                    "transition-all duration-500": slug,
                })}
                width={1300}
                height={630}
            />
        </div>
    );
};

export default CoverImage;
