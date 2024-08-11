"use client";
import { useState, useRef, useEffect } from "react";

export default function FilterBox({ children }: { children: React.ReactNode }) {
  return <div className="py-1 h-full">{children}</div>;
}

export function MockFindName() {
  const [isOpen, setIsOpen] = useState(false);
  const handleToggle = () => setIsOpen((b) => !b);
  return (
    <>
      <MockFindNameButton handleToggle={handleToggle} />
      <MockFindNameOption isOpen={isOpen} handleToggle={handleToggle} />
    </>
  );
}
function MockFindNameOption({
  isOpen,
  handleToggle,
}: {
  isOpen: boolean;
  handleToggle: () => void;
}) {
  const inputValue = useRef("");
  const inputRef = useRef<HTMLInputElement>(null);

  const customHandleToggle = () => {
    alert(inputValue.current);
    handleToggle();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.stopPropagation();
      customHandleToggle();
    }
  };

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);
  return (
    <div
      className={`${isOpen ? "block" : "hidden"} absolute top-[50px] left-0 right-0  bg-white z-10 mx-auto`}
    >
      <div className="px-8 py-4 flex flex-col space-y-2 max-w-[500px] w-full mx-auto">
        <div>모델 이름으로 검색할시 다른 조건은 무시됩니다</div>
        <input
          type="text"
          className=" border py-2 px-4"
          placeholder="모델 이름"
          ref={inputRef}
          onChange={(e) => {
            inputValue.current = e.currentTarget.value;
          }}
          onKeyDown={handleKeyDown}
        />
        <button
          className="bg-black text-white py-3 px-8 text-sm"
          onClick={customHandleToggle}
        >
          검색하기
        </button>
      </div>
    </div>
  );
}

function MockFindNameButton({ handleToggle }: { handleToggle: () => void }) {
  return (
    <button
      className="h-full bg-gray-200 rounded-xl px-2"
      onClick={handleToggle}
    >
      모델 이름 검색
    </button>
  );
}
