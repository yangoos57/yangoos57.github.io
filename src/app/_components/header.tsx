import Link from "next/link";
import Image from "next/image";
const Header = () => {
    return (
        <div className="fixed top-0 h-[55px] bg-nav w-full flex grow items-center z-50">
            <div className="max-w-4xl w-full mx-auto flex justify-between">
                <Link href={"/"} className=" w-full text-white text-base font-medium lg:text-xl px-4">
                    데이터를 종합해 정보를 만듭니다.
                </Link>
                <Link
                    href={"https://github.com/yangoos57"}
                    target="_blank"
                    rel="noreferer"
                    className="text-white md:text-lg px-4 text-right">
                    <span className="hidden md:block">GitHub</span>
                    <Image className="md:hidden" src={"/icon/github.svg"} alt="github" width={30} height={30} />
                </Link>
            </div>
        </div>
    );
};

export default Header;
