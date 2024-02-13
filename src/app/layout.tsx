import Footer from "@/app/_components/footer";
import { PAGE } from "@/lib/constants";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import Script from "next/script";
import "./globals.css";
import Header from "./_components/header";
import GoogleAnalytics from "./_components/ga4/google-analytics-4";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
    metadataBase: new URL(PAGE),
    title: `Yangoos Github Blog`,
    description: `데이터를 종합해 정보를 만듭니다.`,
    openGraph: {
        images: "/og/og-default.png",
    },
    verification: {
        google: "cGIJ_4UB82a6wkT0s71AkXfCqej5-Svsic6s2N0YrLk",
        other: {
            "naver-site-verification": "c3fffb88d53634045cbfe175cb5ecc04afea1ea",
        },
    },
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <GoogleAnalytics GA_TRACKING_ID={process.env.GA_TRACKING_ID!} />
            <head>
                <Script
                    async
                    src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-2341342262931969"
                    crossOrigin="anonymous"
                />
                <link
                    rel="stylesheet"
                    href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
                    integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV"
                    crossOrigin="anonymous"
                />
                <link rel="apple-touch-icon" sizes="180x180" href="/favicon/apple-touch-icon.png" />
                <link rel="icon" type="image/png" sizes="32x32" href="/favicon/favicon-32x32.png" />
                <link rel="icon" type="image/png" sizes="16x16" href="/favicon/favicon-16x16.png" />
                <link rel="manifest" href="/favicon/site.webmanifest" />
                <link rel="mask-icon" href="/favicon/safari-pinned-tab.svg" color="#000000" />
                <link rel="shortcut icon" href="/favicon/favicon.ico" />
                <meta name="msapplication-TileColor" content="#000000" />
                <meta name="msapplication-config" content="/favicon/browserconfig.xml" />
                <meta name="theme-color" content="#000" />
                <link rel="alternate" type="application/rss+xml" href="/feed.xml" />
            </head>
            <body className={inter.className}>
                <Header />
                <div className="pt-[55px] min-h-screen flex flex-col">{children}</div>
                <Footer />
            </body>
        </html>
    );
}
