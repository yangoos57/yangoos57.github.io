type Props = {
  children?: React.ReactNode;
};

const Container = ({ children }: Props) => {
  return <div className="w-full  mx-auto">{children}</div>;
};

export default Container;
