import Link from "next/link";
const Nav = () => {
  return (
    <div className="fixed top-0 h-[55px] bg-nav w-full flex items-center z-50">
      <div className="w-full mx-auto flex justify-between items-center max-w-2xl 2xl:max-w-3xl text-white px-4">
        <Link href={"/"} className="w-full font-medium text-lg md:text-xl  ">
          데이터를 종합해 정보를 만듭니다.
        </Link>
        <div className="text-right space-x-4 md:space-x-8 font-medium  whitespace-nowrap">
          <Link href={"/intro"}>소개</Link>
          <Link
            href={"https://github.com/yangoos57"}
            target="_blank"
            rel="noreferer"
          >
            깃허브
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Nav;
