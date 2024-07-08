import Container from "@/app/_components/container";
import { MoreStories } from "@/app/_components/more-stories";
import { getAllPosts } from "../lib/api";

export default async function Index({ params }: { params: { slug: string } }) {
  const allPosts = getAllPosts();
  const { slug } = params;
  const categories = allPosts.reduce((acc, post) => {
    post.category.forEach((category) => acc.add(category));
    return acc;
  }, new Set<string>());

  const filterName = categories.has(slug as string) ? slug : "all";

  const filteredPosts =
    filterName === "all"
      ? allPosts
      : allPosts.filter((v) => v.category.includes(filterName as string));

  return (
    <main className="relative grow">
      <Container>
        <MoreStories params={filterName as string} posts={filteredPosts} />
      </Container>
    </main>
  );
}
