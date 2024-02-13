import DateFormatter from "./date-formatter";

type Props = {
    title: string;
    date: string;
    category: string[];
};

export function PostHeader({ title, date, category }: Props) {
    return (
        <div className="max-w-2xl 2xl:max-w-3xl mx-auto text-sm lg:text-base pt-12 pb-8 text-gray-500 border-b">
            <div className="mb-6">
                <DateFormatter dateString={date} />
                <div className="flex mb-4">
                    {category.map((c) => (
                        <div className="capitalize pe-2 me-2 text-sm border-e border-nav" key={c}>
                            {c}
                        </div>
                    ))}
                </div>
            </div>
            <h1 className="text-3xl font-bold text-nav">{title}</h1>
        </div>
    );
}
