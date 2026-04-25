"""
文件上传与向量化处理脚本
独立运行，用于将 PDF / Word (.docx) 文件解析、分块并写入向量库。

特性:
  - PDF: 优先使用 PyMuPDF (fitz) 提取文本，支持用 marker 转 Markdown 保留 LaTeX 公式
  - Word: 使用 python-docx 提取文本 + OMML 数学公式转 LaTeX
  - 分块: 使用 LaTeX 公式感知的分块器，避免从中间切断 $...$ 或 $$...$$

用法:
    # 上传单个文件
    python upload_and_index.py /path/to/file.pdf

    # 上传多个文件
    python upload_and_index.py file1.pdf file2.docx file3.pdf

    # 上传整个目录（递归扫描 .pdf 和 .docx）
    python upload_and_index.py /path/to/folder/

    # 指定分块大小
    python upload_and_index.py --chunk-size 512 --chunk-overlap 100 file.pdf

    # 使用 marker 模型做 PDF→Markdown（保留 LaTeX 公式效果最好，需要 GPU）
    python upload_and_index.py --use-marker file.pdf

    # 仅预览解析结果，不写入向量库
    python upload_and_index.py --dry-run file.pdf
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────
# 依赖检查与安装
# ─────────────────────────────────────────────
def _ensure_deps():
    """确保 PDF/Word 解析依赖已安装"""
    missing = []
    try:
        import pypdf  # noqa: F401
    except ImportError:
        missing.append("pypdf")

    try:
        import fitz  # PyMuPDF  # noqa: F401
    except ImportError:
        missing.append("PyMuPDF")

    try:
        import docx2txt  # noqa: F401
    except ImportError:
        missing.append("docx2txt")

    try:
        import docx  # python-docx  # noqa: F401
    except ImportError:
        missing.append("python-docx")

    if missing:
        print(f"📦 正在安装缺失依赖: {', '.join(missing)}")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        print("✅ 依赖安装完成")


_ensure_deps()

# ─────────────────────────────────────────────
# 文件加载器
# ─────────────────────────────────────────────
from langchain_core.documents import Document


# ─────────────────────────────────────────────
# 方案1: PDF — PyMuPDF + marker
# ─────────────────────────────────────────────
def load_pdf_with_fitz(file_path: str) -> list[Document]:
    """使用 PyMuPDF (fitz) 加载 PDF，文本提取质量优于 pypdf"""
    import fitz  # PyMuPDF

    doc = fitz.open(file_path)
    docs = []
    for i, page in enumerate(doc):
        # 优先提取带排版信息的文本
        text = page.get_text("text")
        if text and text.strip():
            docs.append(Document(
                page_content=text.strip(),
                metadata={
                    "source": file_path,
                    "page": i + 1,
                    "total_pages": len(doc),
                    "file_type": "pdf",
                    "extraction_method": "pymupdf",
                }
            ))
    doc.close()
    return docs


def load_pdf_with_marker(file_path: str) -> list[Document]:
    """
    使用 marker 将 PDF 转为 Markdown（保留 LaTeX 公式）。
    需要安装: pip install marker-pdf
    需要 GPU，转换较慢但公式保留效果最好。
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError:
        print("⚠️ marker-pdf 未安装，回退到 PyMuPDF。安装: pip install marker-pdf")
        return load_pdf_with_fitz(file_path)

    print("   ⏳ 使用 marker 转 PDF→Markdown（保留 LaTeX 公式，需要 GPU，较慢）...")
    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)
    rendered = converter(file_path)
    text, _, _ = text_from_rendered(rendered)

    if not text or not text.strip():
        return []

    return [Document(
        page_content=text.strip(),
        metadata={
            "source": file_path,
            "file_type": "pdf",
            "extraction_method": "marker",
            "has_latex": "$$" in text or "\\[" in text,
        }
    )]


def load_pdf(file_path: str, use_marker: bool = False) -> list[Document]:
    """加载 PDF 文件，优先使用 PyMuPDF，可选 marker"""
    if use_marker:
        try:
            return load_pdf_with_marker(file_path)
        except Exception as e:
            print(f"   ⚠️ marker 转换失败: {e}，回退到 PyMuPDF")

    return load_pdf_with_fitz(file_path)


# ─────────────────────────────────────────────
# 方案3: Word — python-docx + OMML→LaTeX
# ─────────────────────────────────────────────
def _omml_to_latex(omml_xml: str) -> str:
    """
    将 Office Math Markup Language (OMML) 转换为 LaTeX。
    
    OMML 是 Word 文档中数学公式的 XML 表示格式。
    此函数覆盖常见的 OMML 元素，将其转为 LaTeX 语法。
    对于无法识别的元素，保留原始 XML 标记。
    """
    from lxml import etree

    # 注册 OMML 命名空间
    nsmap = {
        'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    }

    try:
        root = etree.fromstring(omml_xml)
    except etree.XMLSyntaxError:
        return omml_xml

    def _convert_node(node):
        """递归转换 OMML 节点"""
        tag = etree.QName(node.tag).localname if isinstance(node.tag, str) else ''
        parts = []

        if tag == 'f':  # 分数 <m:f>
            num = ''
            den = ''
            for child in node:
                child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ''
                if child_tag == 'num':
                    num = ''.join(_convert_node(c) for c in child)
                elif child_tag == 'den':
                    den = ''.join(_convert_node(c) for c in child)
            return f'\\frac{{{num}}}{{{den}}}'

        elif tag == 'rad':  # 根号 <m:rad>
            deg = ''
            e = ''
            for child in node:
                child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ''
                if child_tag == 'deg':
                    deg = ''.join(_convert_node(c) for c in child)
                elif child_tag == 'e':
                    e = ''.join(_convert_node(c) for c in child)
            if deg:
                return f'\\sqrt[{deg}]{{{e}}}'
            return f'\\sqrt{{{e}}}'

        elif tag == 'sSup':  # 上标 <m:sSup>
            base = ''
            sup = ''
            for child in node:
                child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ''
                if child_tag == 'e':
                    base = ''.join(_convert_node(c) for c in child)
                elif child_tag == 'sup':
                    sup = ''.join(_convert_node(c) for c in child)
            return f'{base}^{{{sup}}}'

        elif tag == 'sSub':  # 下标 <m:sSub>
            base = ''
            sub = ''
            for child in node:
                child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ''
                if child_tag == 'e':
                    base = ''.join(_convert_node(c) for c in child)
                elif child_tag == 'sub':
                    sub = ''.join(_convert_node(c) for c in child)
            return f'{base}_{{{sub}}}'

        elif tag == 'sSubSup':  # 上下标 <m:sSubSup>
            base = ''
            sub = ''
            sup = ''
            for child in node:
                child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ''
                if child_tag == 'e':
                    base = ''.join(_convert_node(c) for c in child)
                elif child_tag == 'sub':
                    sub = ''.join(_convert_node(c) for c in child)
                elif child_tag == 'sup':
                    sup = ''.join(_convert_node(c) for c in child)
            return f'{base}_{{{sub}}}^{{{sup}}}'

        elif tag == 'nary':  # n-元运算 (求和、积分等) <m:nary>
            lower = ''
            upper = ''
            body = ''
            chr_val = ''
            for child in node:
                child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ''
                if child_tag == 'sub':
                    lower = ''.join(_convert_node(c) for c in child)
                elif child_tag == 'sup':
                    upper = ''.join(_convert_node(c) for c in child)
                elif child_tag == 'e':
                    body = ''.join(_convert_node(c) for c in child)
                elif child_tag == 'naryPr':
                    for prop in child:
                        prop_tag = etree.QName(prop.tag).localname if isinstance(prop.tag, str) else ''
                        if prop_tag == 'chr':
                            chr_val = prop.get(etree.QName('http://schemas.openxmlformats.org/officeDocument/2006/math', 'val'), '')

            # 根据 chr 决定 LaTeX 命令
            nary_map = {
                '∑': '\\sum', '∫': '\\int', '∏': '\\prod',
                '∬': '\\iint', '∮': '\\oint',
            }
            op = nary_map.get(chr_val, '\\sum')
            result = f'{op}'
            if lower:
                result += f'_{{{lower}}}'
            if upper:
                result += f'^{{{upper}}}'
            result += f' {body}'
            return result

        elif tag == 'd':  # 定界符 (括号等) <m:d>
            inner = ''
            for child in node:
                child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ''
                if child_tag == 'e':
                    inner = ''.join(_convert_node(c) for c in child)
            return f'\\left({inner}\\right)'

        elif tag == 'func':  # 函数名 <m:func>
            fname = ''
            body = ''
            for child in node:
                child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ''
                if child_tag == 'fName':
                    fname = ''.join(_convert_node(c) for c in child)
                elif child_tag == 'e':
                    body = ''.join(_convert_node(c) for c in child)
            return f'{fname}\\left({body}\\right)'

        elif tag == 'r':  # 文本运行 <m:r>
            text = ''
            for child in node:
                child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ''
                if child_tag == 't':
                    text += (child.text or '')
            # 处理特殊字符
            text = text.replace('∈', '\\in ').replace('≤', '\\leq ').replace('≥', '\\geq ')
            text = text.replace('≠', '\\neq ').replace('×', '\\times ').replace('÷', '\\div ')
            text = text.replace('±', '\\pm ').replace('→', '\\rightarrow ')
            text = text.replace('∞', '\\infty ').replace('∂', '\\partial ')
            text = text.replace('α', '\\alpha ').replace('β', '\\beta ').replace('γ', '\\gamma ')
            text = text.replace('δ', '\\delta ').replace('θ', '\\theta ').replace('λ', '\\lambda ')
            text = text.replace('μ', '\\mu ').replace('π', '\\pi ').replace('σ', '\\sigma ')
            text = text.replace('ω', '\\omega ').replace('φ', '\\phi ').replace('ψ', '\\psi ')
            return text

        elif tag == 'bar':  # 上划线/下划线 <m:bar>
            body = ''
            position = 'top'
            for child in node:
                child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ''
                if child_tag == 'e':
                    body = ''.join(_convert_node(c) for c in child)
                elif child_tag == 'barPr':
                    for prop in child:
                        prop_tag = etree.QName(prop.tag).localname if isinstance(prop.tag, str) else ''
                        if prop_tag == 'pos':
                            pos_val = prop.get(etree.QName('http://schemas.openxmlformats.org/officeDocument/2006/math', 'val'), 'top')
                            position = pos_val
            if position == 'bot':
                return f'\\underline{{{body}}}'
            return f'\\overline{{{body}}}'

        elif tag == 'acc':  # 重音符号 (hat, tilde, vec 等) <m:acc>
            body = ''
            chr_val = ''
            for child in node:
                child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ''
                if child_tag == 'e':
                    body = ''.join(_convert_node(c) for c in child)
                elif child_tag == 'accPr':
                    for prop in child:
                        prop_tag = etree.QName(prop.tag).localname if isinstance(prop.tag, str) else ''
                        if prop_tag == 'chr':
                            chr_val = prop.get(etree.QName('http://schemas.openxmlformats.org/officeDocument/2006/math', 'val'), '')

            acc_map = {
                '̂': '\\hat', '̃': '\\tilde', '⃗': '\\vec',
                '̄': '\\bar', '̇': '\\dot', '̈': '\\ddot',
                '⏞': '\\overbrace', '⏟': '\\underbrace',
            }
            cmd = acc_map.get(chr_val, '\\hat')
            return f'{cmd}{{{body}}}'

        elif tag == 'eqArr':  # 方程组 <m:eqArr>
            rows = []
            for child in node:
                child_tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ''
                if child_tag == 'e':
                    rows.append(''.join(_convert_node(c) for c in child))
            return ' \\\\ '.join(rows)

        elif tag == 'm':  # 数学区域 (最外层)
            return ''.join(_convert_node(c) for c in node)

        elif tag == 'oMathPara':  # 数学段落
            return ''.join(_convert_node(c) for c in node)

        elif tag == 'oMath':  # 行内/块级数学
            return ''.join(_convert_node(c) for c in node)

        else:
            # 未知节点：递归处理子节点
            if len(node) > 0:
                return ''.join(_convert_node(c) for c in node)
            return node.text or ''

    latex = _convert_node(root)
    return latex.strip()


def load_docx_with_math(file_path: str) -> list[Document]:
    """
    使用 python-docx 加载 Word 文件，提取文本 + OMML 数学公式转 LaTeX。
    
    相比 docx2txt，此方法能：
    1. 识别 Word 中的 OMML 数学公式（<m:oMath> 元素）
    2. 将 OMML 转为 LaTeX 格式，用 $...$ 包裹
    3. 保留公式的精确语义
    """
    from docx import Document as DocxDocument
    from lxml import etree

    doc = DocxDocument(file_path)
    paragraphs_content = []
    math_count = 0

    for para in doc.paragraphs:
        parts = []
        # 遍历段落中的每个子元素
        for child in para._element:
            tag = etree.QName(child.tag).localname if isinstance(child.tag, str) else ''

            if tag == 'r':  # 普通文本运行 <w:r>
                for sub in child:
                    sub_tag = etree.QName(sub.tag).localname if isinstance(sub.tag, str) else ''
                    if sub_tag == 't':
                        text = sub.text or ''
                        parts.append(text)

            elif tag == 'oMathPara':  # 块级数学段落
                for omath in child:
                    omath_tag = etree.QName(omath.tag).localname if isinstance(omath.tag, str) else ''
                    if omath_tag == 'oMath':
                        try:
                            omml_str = etree.tostring(omath, encoding='unicode')
                            latex = _omml_to_latex(omml_str)
                            parts.append(f'\n$$\n{latex}\n$$\n')
                            math_count += 1
                        except Exception as e:
                            # 转换失败时，提取纯文本
                            text = ''.join(omath.itertext())
                            parts.append(f' ${text}$ ')
                            math_count += 1

            elif tag == 'oMath':  # 行内数学公式
                try:
                    omml_str = etree.tostring(child, encoding='unicode')
                    latex = _omml_to_latex(omml_str)
                    parts.append(f' ${latex}$ ')
                    math_count += 1
                except Exception as e:
                    text = ''.join(child.itertext())
                    parts.append(f' ${text}$ ')

        para_text = ''.join(parts).strip()
        if para_text:
            paragraphs_content.append(para_text)

    if not paragraphs_content:
        return []

    full_text = '\n\n'.join(paragraphs_content)
    has_latex = math_count > 0

    if has_latex:
        print(f"   → 提取了 {math_count} 个数学公式 (OMML→LaTeX)")

    return [Document(
        page_content=full_text,
        metadata={
            "source": file_path,
            "file_type": "docx",
            "extraction_method": "python-docx+omml",
            "math_count": math_count,
            "has_latex": has_latex,
        }
    )]


# 兼容旧版 docx2txt（不含公式处理）
def load_docx_simple(file_path: str) -> list[Document]:
    """使用 docx2txt 简单提取（不含公式处理）"""
    import docx2txt

    text = docx2txt.process(file_path)
    if not text or not text.strip():
        return []

    return [Document(
        page_content=text.strip(),
        metadata={
            "source": file_path,
            "file_type": "docx",
            "extraction_method": "docx2txt",
        }
    )]


def load_docx(file_path: str) -> list[Document]:
    """加载 Word 文件，优先使用 OMML→LaTeX 方式"""
    try:
        return load_docx_with_math(file_path)
    except Exception as e:
        print(f"   ⚠️ OMML 提取失败: {e}，回退到 docx2txt")
        return load_docx_simple(file_path)


# 文件类型 → 加载函数 映射
LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
}


def load_file(file_path: str, use_marker: bool = False) -> list[Document]:
    """根据扩展名自动选择加载器"""
    ext = Path(file_path).suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        print(f"⚠️ 不支持的文件类型: {ext}  (支持: {', '.join(LOADERS.keys())})")
        return []
    print(f"📄 加载文件: {file_path}")

    if ext == ".pdf":
        docs = loader(file_path, use_marker=use_marker)
    else:
        docs = loader(file_path)

    # 统计公式信息
    has_latex = any(d.metadata.get("has_latex", False) for d in docs)
    math_total = sum(d.metadata.get("math_count", 0) for d in docs)
    if has_latex or math_total:
        print(f"   → 包含数学公式，LaTeX 格式保留")

    print(f"   → 提取了 {len(docs)} 页/段")
    return docs


def scan_directory(dir_path: str) -> list[str]:
    """递归扫描目录下的 PDF 和 DOCX 文件"""
    files = []
    for root, _, filenames in os.walk(dir_path):
        for fname in filenames:
            if Path(fname).suffix.lower() in LOADERS:
                files.append(os.path.join(root, fname))
    return files


# ─────────────────────────────────────────────
# 方案2: LaTeX 公式感知分块器
# ─────────────────────────────────────────────
class LatexAwareTextSplitter:
    """
    LaTeX 公式感知的文本分块器。
    
    在 RecursiveCharacterTextSplitter 基础上增加：
    1. 将 $$...$$ 和 $...$ 包裹的公式识别为不可分割单元
    2. 分块时优先在公式边界处断开，避免从公式中间截断
    3. 若单个公式超过 chunk_size，整块保留（不做切割）
    """

    # 公式正则：匹配 $$...$$ 和 $...$
    DISPLAY_MATH_RE = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
    INLINE_MATH_RE = re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', re.DOTALL)

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200, **kwargs):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # LaTeX 公式感知的分隔符优先级：
        # 1. 双换行（段落边界）
        # 2. 块级公式边界 $$...$$
        # 3. 单换行
        # 4. 中文句号等标点
        # 5. 行内公式边界 $...$
        # 6. 空格
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=[
                "\n\n",           # 段落
                "$$",             # 块级公式边界
                "\n",             # 换行
                "。", "！", "？",  # 中文标点
                "；", "，",
                "$",              # 行内公式边界
                " ",              # 空格
                "",               # 最后兜底
            ],
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """分块文档，保护公式不被截断"""
        result = []
        for doc in documents:
            text = doc.page_content
            # 先保护公式：用占位符替换，避免被分块器切断
            protected_text, formulas = self._protect_formulas(text)

            # 分块
            splits = self._splitter.split_text(protected_text)

            # 还原公式
            for split_text in splits:
                restored = self._restore_formulas(split_text, formulas)
                # 跳过过短的块
                if len(restored.strip()) < 10:
                    continue
                result.append(Document(
                    page_content=restored.strip(),
                    metadata={**doc.metadata},
                ))

        return result

    def _protect_formulas(self, text: str) -> tuple[str, dict[str, str]]:
        """
        用占位符替换公式，防止分块器从公式中间截断。
        
        Returns:
            (protected_text, formula_map)
            protected_text: 用占位符替换后的文本
            formula_map: 占位符 → 原始公式的映射
        """
        formulas = {}
        counter = [0]

        def replace_display(m):
            key = f"__FORMULA_D{counter[0]}__"
            formulas[key] = m.group(0)  # 保留 $$
            counter[0] += 1
            return key

        def replace_inline(m):
            key = f"__FORMULA_I{counter[0]}__"
            formulas[key] = m.group(0)  # 保留 $
            counter[0] += 1
            return key

        # 先替换块级公式 $$...$$，再替换行内公式 $...$
        protected = self.DISPLAY_MATH_RE.sub(replace_display, text)
        protected = self.INLINE_MATH_RE.sub(replace_inline, protected)

        return protected, formulas

    def _restore_formulas(self, text: str, formulas: dict[str, str]) -> str:
        """将占位符还原为原始公式"""
        for key, formula in formulas.items():
            text = text.replace(key, formula)
        return text


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="上传 PDF/Word 文件到自适应 RAG 向量库（支持数学公式提取）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="文件或目录路径（目录会递归扫描 .pdf 和 .docx）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="分块大小（默认使用 config.py 中的 CHUNK_SIZE）",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="分块重叠大小（默认使用 config.py 中的 CHUNK_OVERLAP）",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Milvus 集合名（默认使用 config.py 中的 COLLECTION_NAME）",
    )
    parser.add_argument(
        "--use-marker",
        action="store_true",
        help="使用 marker 模型做 PDF→Markdown（保留 LaTeX 公式效果最好，需要 GPU）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅解析文件，不写入向量库（用于预览）",
    )
    args = parser.parse_args()

    # ── 1. 收集文件 ────────────────────────
    all_files = []
    for p in args.paths:
        if os.path.isdir(p):
            found = scan_directory(p)
            print(f"📂 扫描目录 {p}: 找到 {len(found)} 个文件")
            all_files.extend(found)
        elif os.path.isfile(p):
            all_files.append(p)
        else:
            print(f"⚠️ 路径不存在: {p}")

    if not all_files:
        print("❌ 没有找到可处理的文件")
        sys.exit(1)

    print(f"\n📋 待处理文件: {len(all_files)} 个")
    for f in all_files:
        print(f"   - {f}")

    # ── 2. 解析文件 ────────────────────────
    print("\n" + "=" * 50)
    print("阶段 1/3: 解析文件（含数学公式提取）")
    print("=" * 50)

    all_docs = []
    for f in all_files:
        try:
            docs = load_file(f, use_marker=args.use_marker)
            all_docs.extend(docs)
        except Exception as e:
            print(f"❌ 解析失败 {f}: {e}")
            import traceback
            traceback.print_exc()

    if not all_docs:
        print("❌ 没有提取到任何文本内容")
        sys.exit(1)

    total_chars = sum(len(d.page_content) for d in all_docs)
    math_docs = [d for d in all_docs if d.metadata.get("has_latex") or d.metadata.get("math_count", 0) > 0]
    print(f"\n✅ 解析完成: {len(all_docs)} 页/段, 共 {total_chars:,} 字符")
    if math_docs:
        total_formulas = sum(d.metadata.get("math_count", 0) for d in math_docs)
        print(f"   📐 其中 {len(math_docs)} 页/段包含数学公式, 共 {total_formulas} 个公式已转为 LaTeX")

    # ── 3. 分块（公式感知）─────────────────
    from config import CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME

    chunk_size = args.chunk_size or CHUNK_SIZE
    chunk_overlap = args.chunk_overlap or CHUNK_OVERLAP
    collection_name = args.collection or COLLECTION_NAME

    print("\n" + "=" * 50)
    print("阶段 2/3: 文本分块（LaTeX 公式感知）")
    print("=" * 50)

    # 检测是否包含 LaTeX 公式，自动选择分块器
    has_any_latex = any(
        d.metadata.get("has_latex") or "$$" in d.page_content or (d.page_content.count("$") >= 2)
        for d in all_docs
    )

    if has_any_latex:
        print("   🔍 检测到 LaTeX 公式，使用公式感知分块器")
        splitter = LatexAwareTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        print("   📝 未检测到公式，使用标准分块器")
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    doc_splits = splitter.split_documents(all_docs)
    print(f"✅ 分块完成: {len(doc_splits)} 个文档块 (chunk_size={chunk_size}, overlap={chunk_overlap})")

    if args.dry_run:
        print("\n🏁 --dry-run 模式，跳过向量库写入。前 3 个文档块预览:")
        for i, doc in enumerate(doc_splits[:3]):
            preview = doc.page_content[:300].replace("\n", "\\n")
            print(f"\n   [{i+1}] source={doc.metadata.get('source','?')} page={doc.metadata.get('page','?')}")
            print(f"       {preview}...")
        return

    # ── 4. 写入向量库 ──────────────────────
    print("\n" + "=" * 50)
    print("阶段 3/3: 写入向量库")
    print("=" * 50)

    from document_processor import DocumentProcessor

    print("⏳ 初始化 DocumentProcessor（加载嵌入模型，可能需要 10-30 秒）...")
    t0 = time.time()
    doc_processor = DocumentProcessor()
    print(f"✅ DocumentProcessor 初始化完成 ({time.time()-t0:.1f}s)")

    if not doc_processor.vectorstore:
        doc_processor.initialize_vectorstore()

    print(f"⏳ 正在向量化并写入 {len(doc_splits)} 个文档块到集合 '{collection_name}'...")
    t1 = time.time()
    doc_processor.add_documents_to_vectorstore(doc_splits)
    elapsed = time.time() - t1
    print(f"✅ 写入完成! ({elapsed:.1f}s, {len(doc_splits)/elapsed:.1f} docs/s)")

    # ── 5. 验证 ────────────────────────────
    print("\n" + "=" * 50)
    print("验证")
    print("=" * 50)
    try:
        retriever = doc_processor.vectorstore.as_retriever(search_kwargs={"k": 2})
        test_query = all_docs[0].page_content[:50]
        results = retriever.invoke(test_query)
        print(f"🔍 用测试查询检索到 {len(results)} 个结果")
        if results:
            print(f"   前 50 字: {results[0].page_content[:50]}...")
    except Exception as e:
        print(f"⚠️ 验证检索失败: {e}")

    print(f"\n🏁 全部完成! 共处理 {len(all_files)} 个文件, 生成 {len(doc_splits)} 个文档块")


if __name__ == "__main__":
    main()
