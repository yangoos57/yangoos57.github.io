"use client";
import markdownStyles from "./markdown-styles.module.css";
import { useEffect } from "react";
import mediumZoom from "medium-zoom";

type Props = {
  content: string;
};

export function PostBody({ content }: Props) {
  useEffect(() => {
    const zoom = mediumZoom(".markdown-body img");

    return () => {
      zoom.detach();
    };
  }, []);

  return (
    <div className=" 2xl:max-w-3xl mx-auto markdown-body">
      <div
        className={markdownStyles["markdown"]}
        dangerouslySetInnerHTML={{ __html: content }}
      />
    </div>
  );
}
