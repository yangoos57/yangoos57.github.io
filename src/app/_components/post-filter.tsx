import Link from "next/link";

const defaultPill = "whitespace-nowrap capitalize rounded-xl px-5 py-2 mx-2 ";
const catPillNotSelected = "bg-main hover:bg-nav hover:text-white";
const catPillSelected = " bg-nav text-white ";

const PostFilter = ({ params, categories }: { params: string; categories: string[] }) => {
    const CatButton = ({ name }: { name: string }) => {
        return (
            <Link href={`/filter/${name.replace(" ", "-")}`}>
                <button className={`${params === name ? catPillSelected : catPillNotSelected} ${defaultPill}`}>
                    {name}
                </button>
            </Link>
        );
    };

    return (
        <div className="sticky top-10 pt-8 pb-4 bg-main z-20 w-full">
            <div className="flex h-[55px] bg-white text-sm font-medium rounded-lg">
                <div className="flex my-auto px-2 border-e">
                    <div className="hidden sm:block pe-4 my-auto">Category</div>
                    <CatButton name="all" />
                </div>
                <div className="flex items-center h-full overflow-auto">
                    <div className="w-full flex no-scrollbar ">
                        {categories.map((v, idx) => {
                            return <CatButton key={idx} name={v} />;
                        })}
                    </div>
                </div>
                <div className="h-[55px] w-[20px] bg-white"></div>
            </div>
        </div>
    );
};

export default PostFilter;
