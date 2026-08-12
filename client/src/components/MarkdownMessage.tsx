import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

// Renders assistant chat replies as markdown (the AI provider returns
// headings/tables/lists) instead of dumping raw "###"/"|...|" syntax as
// plain text. react-markdown renders to React elements, not raw HTML, so
// there's no dangerouslySetInnerHTML / XSS surface here even though the
// content comes from an external provider.
const components: Components = {
  p: ({ children }) => <p className="mb-2 leading-snug last:mb-0">{children}</p>,
  h1: ({ children }) => <h1 className="mb-1.5 mt-2.5 text-sm font-semibold first:mt-0">{children}</h1>,
  h2: ({ children }) => <h2 className="mb-1.5 mt-2.5 text-sm font-semibold first:mt-0">{children}</h2>,
  h3: ({ children }) => <h3 className="mb-1 mt-2 text-sm font-semibold first:mt-0">{children}</h3>,
  ul: ({ children }) => <ul className="mb-2 list-disc space-y-0.5 pl-4 last:mb-0">{children}</ul>,
  ol: ({ children }) => <ol className="mb-2 list-decimal space-y-0.5 pl-4 last:mb-0">{children}</ol>,
  li: ({ children }) => <li className="leading-snug">{children}</li>,
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent underline underline-offset-2">
      {children}
    </a>
  ),
  strong: ({ children }) => <strong className="font-semibold text-fg">{children}</strong>,
  code: ({ className, children }) => {
    // remark marks fenced code blocks with a "language-xxx" className on the
    // inner <code>; plain `inline code` has none -- that's the signal used
    // elsewhere in this file to size/pad the two cases differently.
    const isBlock = Boolean(className);
    return isBlock ? (
      <code className="block font-mono text-xs">{children}</code>
    ) : (
      <code className="rounded bg-surface px-1 py-0.5 font-mono text-xs">{children}</code>
    );
  },
  pre: ({ children }) => (
    <pre className="mb-2 overflow-x-auto rounded-md border border-border bg-surface p-2 last:mb-0">{children}</pre>
  ),
  table: ({ children }) => (
    <div className="mb-2 overflow-x-auto last:mb-0">
      <table className="w-full border-collapse text-xs">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead>{children}</thead>,
  th: ({ children }) => (
    <th className="border border-border px-1.5 py-1 text-left font-semibold text-fg">{children}</th>
  ),
  td: ({ children }) => <td className="border border-border px-1.5 py-1 align-top">{children}</td>,
  blockquote: ({ children }) => (
    <blockquote className="mb-2 border-l-2 border-border pl-2 text-muted last:mb-0">{children}</blockquote>
  ),
  hr: () => <hr className="my-2 border-border" />,
};

export function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="text-sm">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
