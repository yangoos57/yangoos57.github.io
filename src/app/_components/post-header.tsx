import DateFormatter from "./date-formatter";

type Props = {
  title: string;
  date: string;
  category: string[];
};

export function PostHeader({ title, date, category }: Props) {
  return (
    <div className="mx-auto py-8 border-b">
      <div className="flex">
        <div className="flex mb-4">
          {category.map((c) => (
            <div className="capitalize pe-2 me-2 text-nav" key={c}>
              {c}
            </div>
          ))}
          <DateFormatter dateString={date} />
        </div>
      </div>
      <h1 className="text-2xl md:text-3xl font-bold text-nav">{title}</h1>
    </div>
  );
}
