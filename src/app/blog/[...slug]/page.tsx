import { Metadata } from "next";
import { notFound } from "next/navigation";
import { getPostSlugs, getPostBySlug } from "../../../lib/api";
import markdownToHtml from "../../../lib/markdownToHtml";

import Container from "../../_components/container";
import { PostBody } from "../../_components/post-body";
import { PostHeader } from "../../_components/post-header";
import { join } from "path";

export default async function Post({ params }: { params: { slug: string[] } }) {
    const post = getPostBySlug(join(...params.slug));

    if (!post) {
        return notFound();
    }

    const content = await markdownToHtml(post.content || "");

    return (
        <main className="mx-auto w-full whitespace-wrap">
            <Container>
                <article className=" mb-32">
                    <PostHeader title={post.title} date={post.date} category={post.category} />
                    <PostBody content={content} />
                </article>
            </Container>
        </main>
    );
}

export async function generateMetadata({ params }: { params: { slug: string[] } }) {
    const post = getPostBySlug(join(...params.slug));

    if (!post) {
        return notFound();
    }

    const title = `${post.title}`;

    return {
        openGraph: {
            title,
            images: [post.ogImage.url],
        },
    };
}

export async function generateStaticParams() {
    const slugs = getPostSlugs();

    return slugs.map((post) => ({
        slug: post.split("/"),
    }));
}
