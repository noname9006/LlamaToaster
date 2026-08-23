import { describe, expect, it } from "vitest";
import { parseNextLink, parseNextCursor, parseQuant } from "./hf.js";

describe("parseQuant", () => {
  it("extracts a plain quant code", () => {
    expect(parseQuant("Llama-3.1-8B-Instruct-Q4_K_M.gguf")).toBe("Q4_K_M");
  });

  it("extracts an IQ code", () => {
    expect(parseQuant("model-IQ2_XS.gguf")).toBe("IQ2_XS");
  });

  it("prefixes an unsloth Dynamic quant with UD-", () => {
    expect(parseQuant("gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf")).toBe("UD-Q4_K_XL");
  });

  it("handles an underscore-separated UD prefix too", () => {
    expect(parseQuant("model_UD_IQ2_M.gguf")).toBe("UD-IQ2_M");
  });

  it("returns null when no quant token is present", () => {
    expect(parseQuant("some-random-file.gguf")).toBeNull();
  });
});

describe("parseNextLink", () => {
  it("extracts the full next URL from a Link header", () => {
    const link =
      '<https://huggingface.co/api/models/x/tree/main?recursive=true&cursor=abc>; rel="next"';
    expect(parseNextLink(link)).toBe(
      "https://huggingface.co/api/models/x/tree/main?recursive=true&cursor=abc"
    );
  });

  it("returns null for a missing header (last page)", () => {
    expect(parseNextLink(null)).toBeNull();
  });

  it("returns null when the header has no rel=\"next\" link", () => {
    expect(parseNextLink('<https://huggingface.co/api/models?x=1>; rel="prev"')).toBeNull();
  });

  it("returns null for a malformed/unparseable URL", () => {
    expect(parseNextLink("<not a url>; rel=\"next\"")).toBeNull();
  });
});

describe("parseNextCursor", () => {
  it("extracts the cursor query param from a real HF Link header", () => {
    const link =
      '<https://huggingface.co/api/models?search=llama&filter=gguf&limit=5&sort=downloads&direction=-1&cursor=eyIkb3IiOlt7ImRvd25sb2FkcyI6MTg2MjY5fV19>; rel="next"';
    expect(parseNextCursor(link)).toBe(
      "eyIkb3IiOlt7ImRvd25sb2FkcyI6MTg2MjY5fV19"
    );
  });

  it("returns null for a missing header (last page)", () => {
    expect(parseNextCursor(null)).toBeNull();
  });

  it("returns null when the header has no rel=\"next\" link", () => {
    expect(parseNextCursor('<https://huggingface.co/api/models?x=1>; rel="prev"')).toBeNull();
  });

  it("returns null for a malformed/unparseable URL", () => {
    expect(parseNextCursor('<not a url>; rel="next"')).toBeNull();
  });

  it("returns null when the next link has no cursor param", () => {
    expect(parseNextCursor('<https://huggingface.co/api/models?search=llama>; rel="next"')).toBeNull();
  });
});
