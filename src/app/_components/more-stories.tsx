import { Post } from "@/interfaces/post";
import { PostPreview } from "./post-preview";

type Props = {
  posts: Post[];
  params: string;
};

export function MoreStories({ posts, params }: Props) {
  return (
    <section>
      <div className="pt-12 pb-4 text-2xl font-semibold capitalize">
        <span>블로그 글</span>
        <span className="text-xl px-1">
          ({posts.filter((r) => r.publish).length})
        </span>
      </div>
      <div className="py-2 space-y-8">
        {posts.map(
          (post) =>
            post.publish && (
              <PostPreview
                key={post.slug}
                title={post.title}
                thumbnail={post.thumbnail}
                date={post.date}
                category={post.category}
                slug={post.slug}
                desc={post.desc}
              />
            )
        )}
      </div>
    </section>
  );
}
