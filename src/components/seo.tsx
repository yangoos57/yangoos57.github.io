import React from "react"
import { Helmet } from "react-helmet"

import useSiteMetadata from "Hooks/useSiteMetadata"
import type { MarkdownRemarkFrontmatter } from "Types/GraphQL"
import defaultOpenGraphImage from "../images/og-default.png"

const DEFAULT_LANG = "kr"

type Meta = React.DetailedHTMLProps<
  React.MetaHTMLAttributes<HTMLMetaElement>,
  HTMLMetaElement
>[]

interface SEOProps extends Pick<MarkdownRemarkFrontmatter, "title" | "desc"> {
  image?: string
  meta?: Meta
}

const SEO: React.FC<SEOProps> = ({ title, desc = "", image }) => {
  const site = useSiteMetadata()
  const description = desc || site.description
  const ogImageUrl =
    site.siteUrl ?? "" + (image || (defaultOpenGraphImage as string))

  return (
    <Helmet
      htmlAttributes={{ lang: site.lang ?? DEFAULT_LANG }}
      title={title ?? ""}
      titleTemplate={`%s | ${site.title}`}
      meta={
        [
          {
            name :'google-site-verification',
            content: "cGIJ_4UB82a6wkT0s71AkXfCqej5-Svsic6s2N0YrLk"
          },
          {
            name :'naver-site-verification',
            content: "c3fffb88d53634045cbfe175cb5ecc04afea1ea0"
          },
          {
            name: "description",
            content: description,
          },
          {
            property: "og:title",
            content: title,
          },
          {
            property: "og:description",
            content: description,
          },
          {
            property: "og:type",
            content: "website",
          },
          {
            name: "twitter:card",
            content: "summary",
          },
          {
            name: "twitter:creator",
            content: site.author,
          },
          {
            name: "twitter:title",
            content: title,
          },
          {
            name: "twitter:description",
            content: description,
          },
          {
            property: "image",
            content: ogImageUrl,
          },
          {
            property: "og:image",
            content: ogImageUrl,
          },
          {
            property: "twitter:image",
            content: ogImageUrl,
          },
        ] as Meta
      }
    />
  )
}

export default SEO
