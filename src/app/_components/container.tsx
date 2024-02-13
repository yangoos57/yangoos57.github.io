type Props = {
    children?: React.ReactNode;
};

const Container = ({ children }: Props) => {
    return <div className="w-full max-w-4xl mx-auto px-4">{children}</div>;
};

export default Container;
