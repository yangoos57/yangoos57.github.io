import { Post } from "@/interfaces/post";
import { PostPreview } from "./post-preview";

type Props = {
    posts: Post[];
    params: string;
};

export function MoreStories({ posts, params }: Props) {
    return (
        <section>
            <div className="pt-12 pb-4 text-3xl sm:text-4xl font-semibold capitalize">
                {params}
                <span className="text-2xl sm:text-3xl">({posts.length})</span>
            </div>
            <div className="py-2 grid grid-cols-1 md:grid-cols-2 md:gap-x-8 lg:gap-x-16 gap-y-10 md:gap-y-16 pb-16">
                {posts.map((post) => (
                    <PostPreview
                        key={post.slug}
                        title={post.title}
                        thumbnail={post.thumbnail}
                        date={post.date}
                        category={post.category}
                        slug={post.slug}
                        desc={post.desc}
                    />
                ))}
            </div>
        </section>
    );
}
