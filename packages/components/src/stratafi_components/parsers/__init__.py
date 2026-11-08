"""The parsers: Markdown and plain text, text-layer PDF readers, layout
parsers (docling, unstructured), markitdown, office formats, Tika, OCR and hosted adapters."""

from stratafi_core.components import Component


def components() -> list[type[Component]]:
    from stratafi_components.parsers import hosted, layout, markdown, ocr, office, pdf, tika

    return [
        *markdown.COMPONENTS,
        *pdf.COMPONENTS,
        *layout.COMPONENTS,
        *office.COMPONENTS,
        *tika.COMPONENTS,
        *ocr.COMPONENTS,
        *hosted.COMPONENTS,
    ]
