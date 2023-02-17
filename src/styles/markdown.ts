import styled from "styled-components"
import type typography from "./typography"

const Markdown = styled.article<{ rhythm: typeof typography["rhythm"] }>`
  h1,
  h2,
  h3,
  h4,
  h5,
  h6 {
    font-weight: var(--font-weight-bold);
    color: var(--color-highlight) !important;
  }

  td,
  th {
    border-bottom: 1px solid var(--color-gray-3);
  }

  strong {
    font-weight: var(--font-weight-semi-bold);
  }

  a,
  p {
    font-weight: var(--font-weight-regular);

    @media (max-width: ${({ theme }) => theme.device.lg}) {
      font-size: 0.95rem !important;
    }
  
  }

  a {
    text-decoration: underline;
    color: var(--color-blue) !important;
    * {
      color: var(--color-blue) !important;
    }
    &:hover,
    &:active {
      text-decoration: underline;
    }
  }

  & > *:first-child {
    margin-top: 0;
  }

  h1 {
    font-size: 2.5rem;
    @media (max-width: ${({ theme }) => theme.device.sm}) {
      font-size: 2rem;
    }
  }

  h2 {
    font-size: 1.75rem;
    line-height: 1.3;
    margin-bottom: ${({ rhythm }) => rhythm(1)};
    margin-top: ${({ rhythm }) => rhythm(2.25)};

    @media (max-width: ${({ theme }) => theme.device.sm}) {
      font-size: 1.3125rem;
    }
  }

  h3 {
    font-size: 1.31951rem;
    line-height: 1.3;
    margin-bottom: ${({ rhythm }) => rhythm(1)};
    margin-top: ${({ rhythm }) => rhythm(1.5)};

    @media (max-width: ${({ theme }) => theme.device.sm}) {
      font-size: 1.1875rem;
    }
  }

  h4,
  h5,
  h6 {
    font-size : 1.1rem;
    margin-bottom: ${({ rhythm }) => rhythm(0.5)};
    margin-top: ${({ rhythm }) => rhythm(1)};
  }

  ul,
  ol {
    margin-top: ${({ rhythm }) => rhythm(0.5)};
    margin-bottom: ${({ rhythm }) => rhythm(1)};
    margin-left: ${({ rhythm }) => rhythm(1.25)};
    
  }

  li > ul,
  li > ol {
    margin-bottom: 0;
  }

  li > p {
    margin-top: 10px;
    margin-bottom: 0;
  }

  li > ol,
  li > ul {
    margin-left: ${({ rhythm }) => rhythm(1.25)};
  }

  li {
    margin-bottom: ${({ rhythm }) => rhythm(0.3)};

    @media (max-width: ${({ theme }) => theme.device.lg}) {
      font-size: 0.95rem !important;
    }
  }

  p,
  li,
  blockquote {
    font-size: 1rem;
  }

  p {
    line-height: 1.68;
    text-align: left;
    margin-bottom: var(--sizing-lg);
  }

  hr {
    margin: var(--sizing-lg) 0;
    background: var(--color-gray-3);
  }

  blockquote {
    
    border-left: 0.25rem solid var(--color-blockquote-1);
    padding-left: var(--sizing-base);
    margin: var(--sizing-md) 0;
    * {
      color: var(--color-blockquote);
    }
  }

  img {
    display: block;    
  }
  
  figcaption {
    text-align : center;
    color : var(--color-gray-4);
    font-size : 0.98rem;
  }

  pre,
  code {
    font-family: SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace;
    background-color: var(--color-code-block);
    font-size : var(--text-sm);
  }

  pre {
    border: 1px solid var(--color-gray-3);
  }

  pre.grvsc-container {
    margin: var(--sizing-md) 0;
  }
  
  .grvsc-line-highlighted::before {
    background-color: var(--color-code-highlight) !important;
    box-shadow: inset 4px 0 0 0 var(--color-code-highlight-border) !important;
  }

  *:not(pre) > code {
    background-color: var(--color-code);
    color: var(--color-code-font);
    padding: 0.2rem 0.4rem;
    margin: 0;
    font-size: 85%;
    font-weight: 500;
    border-radius: 3px;
  }
  li {
    list-style: "▸ " !important;
    line-height:30px !important;
    margin-bottom : 20px !important;
  }
  li > ul > li{
    list-style: "- " !important;
    font-size:0.95rem !important;
    margin-bottom : 10px !important;
  }
  
`

export default Markdown
