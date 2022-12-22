/**
 * @typedef {Object} Links
 * @prop {string} github Your github repository
 */

/**
 * @typedef {Object} MetaConfig
 * @prop {string} title Your website title
 * @prop {string} description Your website description
 * @prop {string} author Maybe your name
 * @prop {string} siteUrl Your website URL
 * @prop {string} lang Your website Language
 * @prop {string} utterances Github repository to store comments
 * @prop {Links} links
 * @prop {string} favicon Favicon Path
 */

/** @type {MetaConfig} */
const metaConfig = {
  title: "목적에 맞는 데이터를 취합해 정보를 만듭니다.",
  description: `공부한 자료를 복습하기 편하게 블로그 형태로 정리했습니다.`,
  author: "Yangoos",
  siteUrl: "https://yangoos57.github.io",
  lang: "kr",
  utterances: "yangoos57/leeway.github.io",
  links: {
    github: "https://github.com/yangoos57",
  },
  favicon: "src/images/icon.webp",
}

// eslint-disable-next-line no-undef
module.exports = metaConfig
