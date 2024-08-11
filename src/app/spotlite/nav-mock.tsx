import Logo from "./logo";
import FilterBox, { MockFindName } from "./mock-find-name";

const NavMock = () => {
  return (
    <div className="absolute top-0 left-0 h-[50px] shadow-xl w-full flex items-center justify-between px-2 sm:px-8 ">
      <Logo />
      <FilterBox>
        <MockFindName />
      </FilterBox>
      <div>로그인</div>
    </div>
  );
};

export default NavMock;
