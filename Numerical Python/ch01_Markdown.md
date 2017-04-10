
# IPython notebook Markdown summary
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [IPython notebook Markdown summary](#ipython-notebook-markdown-summary)
  * [Italics](#italics)
  * [Bold](#bold)
  * [Strike-through](#strike-through)
  * [Fixed-width font](#fixed-width-font)
  * [URLs](#urls)
  * [New paragraph](#new-paragraph)
  * [Verbatim](#verbatim)
  * [Table](#table)
  * [Horizontal line](#horizontal-line)
  * [Headings](#headings)
* [Level 1 heading](#level-1-heading)
  * [Level 2 heading](#level-2-heading)
    * [Level 3 heading](#level-3-heading)
  * [Block quote](#block-quote)
  * [Unordered list](#unordered-list)
  * [Ordered list](#ordered-list)
  * [Image](#image)
  * [LaTeX](#latex)
  * [Versions](#versions)

<!-- tocstop -->


Robert Johansson

Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).

The source code listings can be downloaded from http://www.apress.com/9781484205549

Test notebook for Markdown table in Chapter 1.

## Italics

Text that is surrounded by asterisks `*text*` is displayed as italics: *text*

## Bold

Text that is surrounded by double asterisks `**text**` is displayed as bold: **text**

## Strike-through

Text that is surrounded by double tidle `~~text~~` is displayed as strike-through: ~~text~~

## Fixed-width font

Text that is quoted with ` characters are displayed as fixed-width font:

`text`

## URLs

URLs are written as	`[URL text](http://www.example.com)`: [URL text](http://www.example.com)

## New paragraph

Separate the text of two paragraphs with an empty line.

This is a new paragraph.

## Verbatim

Text that start with four spaces is displayed as verbatim:

    def func(x):
        return x ** 2

## Table

The format for tables are as follows:

    | A | B | C |
    |---|---|---|
    | 1 | 2 | 3 |
    | 4 | 5 | 6 |

| A | B | C |
|---|---|---|
| 1 | 2 | 3 |
| 4 | 5 | 6 |

## Horizontal line

A line with three dashes `---` is shown as a horizontal line:

---

## Headings

Lines starting with one `#` is a heading level 1, `##` is heading level 2, `###` is heading level 3, etc.

# Level 1 heading
## Level 2 heading
### Level 3 heading

## Block quote

Lines that start with `>` are displayed as a block quote:

> Text here is indented and offset
> from the main text body.

## Unordered list

Unordered lists are created by starting lines with `*`

* Item one
* Item two
* Item three

## Ordered list

Ordered lists are created by simply enumerating lines with numbers followed a period: 1. ..., 2. ..., etc.
1. Item one
2. Item two
3. Item three

## Image

Images can be included using `![Alternative text](image-file.png)` or `![Alternative text](http://www.example.com/image.png)`:

![Alternative text](image-file.png)

![Alternative text](http://www.example.com/image.png)

## LaTeX

Inline LaTeX equations can be included using `$\LaTeX$`: $\LaTeX$

Displayed LaTeX equations (centered, and on a new line): `$$\LaTeX$$`

$$\LaTeX$$

It is also possible to use latex environments like equation, eqnarray, align:

`\begin{equation} x = 1 \end{equation}`

\begin{equation} x = 1 \end{equation}

`\begin{eqnarray} x = 2 \end{eqnarray}`

\begin{eqnarray} x = 2 \end{eqnarray}

`\begin{align} x = 3 \end{align}`

\begin{align} x = 3 \end{align}


## Versions


```python
%reload_ext version_information
```


```python
%version_information
```




<table><tr><th>Software</th><th>Version</th></tr><tr><td>Python</td><td>2.7.10 64bit [GCC 4.2.1 (Apple Inc. build 5577)]</td></tr><tr><td>IPython</td><td>4.0.0</td></tr><tr><td>OS</td><td>Darwin 14.5.0 x86_64 i386 64bit</td></tr></table>
