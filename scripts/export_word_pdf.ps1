param(
    [string]$DocumentPath = 'output/docx/object_attention_paper.docx',
    [string]$PdfPath = 'output/pdf/object_attention_paper.pdf'
)
$ErrorActionPreference = 'Stop'
$researchDoc = (Resolve-Path -LiteralPath $DocumentPath).Path
$researchPdf = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $PdfPath))
$researchWord = $null
$researchOpenDoc = $null
try {
    $researchWord = New-Object -ComObject Word.Application
    $researchWord.Visible = $false
    $researchWord.DisplayAlerts = 0
    $researchOpenDoc = $researchWord.Documents.Open($researchDoc, $false, $true, $false)
    $researchOpenDoc.ExportAsFixedFormat($researchPdf, 17)
    Write-Output $researchPdf
} finally {
    if ($null -ne $researchOpenDoc) { $researchOpenDoc.Close(0) }
    if ($null -ne $researchWord) { $researchWord.Quit() }
}
