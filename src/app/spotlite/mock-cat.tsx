function CatButton({ name }: { name: string }) {
  return (
    <button className="rounded-full py-2 px-6 my-1 bg-black/90 text-white text-xs text-nowrap">
      {name}
    </button>
  );
}

export function MockCats() {
  const cats = [
    "FW 룩북은 딱 이친구",
    "멋쟁이 시니어 모델",
    "언더웨어 가능 모델",
    "FW 룩북은 딱 이친구",
    "멋쟁이 시니어 모델",
    "가능 모델",
    "FW  딱 이친구",
    "멋쟁이 시니어 모델",
    "언더웨어 가능 모델",
    "FW 룩북은 딱 이친구",
    "멋쟁이 시니어 모델",
    "언더웨어 가능 모델",
  ];
  return cats.map((v) => <CatButton name={v} />);
}

export default function MockCatsBox({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex pb-4 sm:pb-0 sm:flex-wrap overflow-x-scroll sm:justify-center space-x-4">
      {children}
    </div>
  );
}
