"""Build the Week 1 IEEE-CIS EDA Word report from the saved EDA summary and charts."""
from __future__ import annotations
import json
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
SUMMARY=json.loads((ROOT/'eda_summary.json').read_text(encoding='utf-8'))
DOCX=ROOT/'Week1_IEEE_CIS_Fraud_EDA_Report.docx'
NAVY='18324B'; BLUE='3977A8'; TEAL='1F7A8C'; PALE='EAF2F8'; LIGHT='F4F7FA'; GRAY='536575'

def shade(cell,fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd');shd.set(qn('w:fill'),fill);tcPr.append(shd)

def set_cell_text(cell,text,bold=False,color=NAVY,size=9):
    cell.text='';p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(0)
    r=p.add_run(str(text));r.bold=bold;r.font.name='Aptos';r.font.size=Pt(size);r.font.color.rgb=RGBColor.from_string(color)
    cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER

def add_table(doc,headers,rows,widths=None):
    table=doc.add_table(rows=1,cols=len(headers));table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.style='Light Shading Accent 1'
    for i,h in enumerate(headers):shade(table.rows[0].cells[i],NAVY);set_cell_text(table.rows[0].cells[i],h,True,'FFFFFF',9)
    for ridx,row in enumerate(rows):
        cells=table.add_row().cells
        for i,v in enumerate(row):
            if ridx%2==1:shade(cells[i],LIGHT)
            set_cell_text(cells[i],v,color=NAVY,size=9)
    if widths:
        for row in table.rows:
            for i,w in enumerate(widths):row.cells[i].width=Inches(w)
    doc.add_paragraph().paragraph_format.space_after=Pt(2)
    return table

def para(doc,text='',style=None,bold_lead=None):
    p=doc.add_paragraph(style=style) if style else doc.add_paragraph()
    p.paragraph_format.space_after=Pt(6);p.paragraph_format.line_spacing=1.08
    if bold_lead and text.startswith(bold_lead):
        p.add_run(bold_lead).bold=True;p.add_run(text[len(bold_lead):])
    else:p.add_run(text)
    return p

def bullet(doc,text):
    p=doc.add_paragraph(style='List Bullet');p.paragraph_format.space_after=Pt(4);p.paragraph_format.line_spacing=1.05;p.add_run(text);return p

def add_figure_page(doc,number,title,filename,explanation,why,alt):
    doc.add_page_break()
    doc.add_heading(title,level=1)
    para(doc,explanation)
    pic=doc.add_picture(str(ROOT/'charts'/filename),width=Inches(6.45))
    pic._inline.docPr.set('descr',alt)
    p=doc.paragraphs[-1];p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(3)
    cap=doc.add_paragraph();cap.alignment=WD_ALIGN_PARAGRAPH.CENTER;cap.paragraph_format.space_after=Pt(8)
    run=cap.add_run(f'Figure {number}. {title}.');run.italic=True;run.font.size=Pt(9);run.font.color.rgb=RGBColor.from_string(GRAY)
    p=para(doc,why);p.paragraph_format.left_indent=Inches(.18);p.paragraph_format.right_indent=Inches(.18)

def main():
    doc=Document();section=doc.sections[0];section.top_margin=Inches(.68);section.bottom_margin=Inches(.68);section.left_margin=Inches(.78);section.right_margin=Inches(.78)
    normal=doc.styles['Normal'];normal.font.name='Aptos';normal.font.size=Pt(10.5);normal.font.color.rgb=RGBColor.from_string(NAVY)
    normal.paragraph_format.space_after=Pt(6)
    for sty,size,color in [('Title',28,NAVY),('Heading 1',17,NAVY),('Heading 2',12,TEAL)]:
        s=doc.styles[sty];s.font.name='Aptos Display' if sty!='Normal' else 'Aptos';s.font.size=Pt(size);s.font.bold=True;s.font.color.rgb=RGBColor.from_string(color)
        s.paragraph_format.space_before=Pt(9);s.paragraph_format.space_after=Pt(5)
    doc.core_properties.title='Week 1 Exploratory Data Analysis IEEE CIS Fraud Detection'
    doc.core_properties.subject='Training-only exploratory data analysis for the IEEE-CIS Fraud Detection project'

    p=doc.add_paragraph(style='Title');p.paragraph_format.space_after=Pt(2);p.add_run('Week 1 Exploratory Data Analysis')
    p=doc.add_paragraph();p.paragraph_format.space_after=Pt(16);r=p.add_run('IEEE CIS Fraud Detection');r.font.size=Pt(15);r.font.color.rgb=RGBColor.from_string(TEAL)
    call=doc.add_table(rows=1,cols=1);call.alignment=WD_TABLE_ALIGNMENT.CENTER;shade(call.cell(0,0),PALE)
    set_cell_text(call.cell(0,0),f"The cleaned training table contains {SUMMARY['rows']:,} transactions, with {SUMMARY['target_counts']['1']:,} fraud cases ({SUMMARY['fraud_rate']:.2%}). The analysis also finds a long-tailed transaction amount distribution and substantial missingness in identity fields. Product and card categories have different observed fraud rates; these are descriptive patterns and do not establish cause.",False,NAVY,11)
    doc.add_paragraph()

    doc.add_heading('Project Objective',level=1)
    para(doc,'This Week 1 analysis describes fraud prevalence, transaction amounts, missing values, and selected categorical patterns in the IEEE-CIS training data. It provides a clear baseline for later project work without using test data or training a model.')
    doc.add_heading('Dataset Description',level=1)
    para(doc,'The input is the cleaned training transaction table joined to training identity data by TransactionID. Each transaction remains one row; identity fields are missing when no linked identity row or source value is available. The analysis uses the preserved isFraud target.')
    add_table(doc,['Dataset','Rows','Columns','Use'],[
      ['Merged training transactions and identity',f"{SUMMARY['rows']:,}",'434','EDA input; includes isFraud'],
      ['Training target - legitimate',f"{SUMMARY['target_counts']['0']:,}",'—','isFraud = 0'],
      ['Training target - fraudulent',f"{SUMMARY['target_counts']['1']:,}",'—','isFraud = 1'],
    ],[3.0,1.0,1.0,1.45])
    para(doc,'Input file: '+str((PROJECT/'cleaned_data'/'train_cleaned.csv').resolve()))
    para(doc,'No test dataset or test labels were opened for this analysis.')

    doc.add_page_break()
    doc.add_heading('Data Cleaning Methodology',level=1)
    bullet(doc,'The analysis reads the previously cleaned training transaction and identity merge. It does not alter or overwrite any source or cleaned dataset.')
    bullet(doc,'TransactionID was used for the earlier one-to-one left join. The cleaned training table preserves all 590,540 transaction rows and the original isFraud target.')
    bullet(doc,'No duplicate rows were removed during the earlier cleaning, and columns with high missingness were retained.')
    bullet(doc,'Numeric-looking features were kept in their recorded numeric types. ProductCD, card4, card6, and email domain fields were treated as categories for grouped summaries.')

    doc.add_heading('Missing Value Treatment Decisions',level=1)
    para(doc,'For EDA, missing values were kept as missing so the analysis could describe their extent and location. No column was dropped and no value was imputed for the charts or statistics. In the separate model-preparation output, numeric imputation uses medians fitted on training-fit rows only; categorical values use a dedicated missing code and an unseen-category code. Those transformations were not used to calculate the EDA findings.')
    stats=SUMMARY['missingness']
    bullet(doc,f"Across the merged training table, the average missing rate is {stats['average_identity_feature_pct']:.1f}% across identity fields and {stats['average_transaction_feature_pct']:.1f}% across transaction features.")
    bullet(doc,f"The most-missing field is {stats['top_20'][0]['column']} ({stats['top_20'][0]['missing_pct']:.2f}% missing). Several identity fields exceed 99% missingness, so their potential value should be assessed before any later feature removal.")

    doc.add_heading('EDA Methodology',level=1)
    bullet(doc,'Counted the full target distribution and calculated class percentages.')
    bullet(doc,'Compared TransactionAmt summaries by target and plotted the positive amounts on a log10 scale to make the long tail visible.')
    bullet(doc,'Calculated missing counts and percentages for every column in the merged training table.')
    bullet(doc,'Calculated ProductCD, card4, and card6 fraud rates from grouped counts. ProductCD and card category charts include Wilson 95% intervals; card charts show categories with at least 1,000 rows for readability.')
    bullet(doc,'All category-rate comparisons are descriptive. They do not show that a category causes fraud or account for other variables.')

    doc.add_page_break();doc.add_heading('Findings and Insights',level=1)
    amt=SUMMARY['amount_summary'];prod={x['category']:x for x in SUMMARY['productcd']};cards4={x['category']:x for x in SUMMARY['card4']};cards6={x['category']:x for x in SUMMARY['card6']}
    insight_texts=[
      f"Fraud is uncommon in the labeled training data: {SUMMARY['target_counts']['1']:,} of {SUMMARY['rows']:,} rows are fraudulent ({SUMMARY['fraud_rate']:.2%}); the remaining {SUMMARY['target_counts']['0']:,} rows are legitimate ({1-SUMMARY['fraud_rate']:.2%}). This imbalance matters when interpreting later model metrics.",
      f"TransactionAmt has a long right tail. Its mean is {amt['overall']['mean']:.2f}, above its median of {amt['overall']['median']:.2f}; the 99th percentile is {amt['overall']['p99']:.2f}, while the maximum is {amt['overall']['max']:.2f}.",
      f"The fraudulent group has a median transaction amount of {amt['by_class']['1']['median']:.2f}, compared with {amt['by_class']['0']['median']:.2f} for legitimate rows. Its 90th percentile is {amt['by_class']['1']['p90']:.2f} versus {amt['by_class']['0']['p90']:.2f}. These summaries describe the sample and do not explain why the amounts differ.",
      f"ProductCD C has the highest observed fraud rate at {prod['C']['fraud_rate']:.2%} across {prod['C']['total']:,} rows. ProductCD W has the most rows ({prod['W']['total']:,}) and an observed rate of {prod['W']['fraud_rate']:.2%}. Category rates should be read with their sample sizes and intervals.",
      f"Missingness is concentrated in identity fields: their average missing rate is {stats['average_identity_feature_pct']:.1f}%, versus {stats['average_transaction_feature_pct']:.1f}% for transaction features. id_24 is missing in {stats['top_20'][0]['missing_count']:,} rows ({stats['top_20'][0]['missing_pct']:.2f}%).",
      f"At least one joined identity field is present for {SUMMARY['identity_presence_by_class']['1']['present']:,} of {SUMMARY['target_counts']['1']:,} fraud rows ({SUMMARY['identity_presence_by_class']['1']['present']/SUMMARY['target_counts']['1']:.2%}) and {SUMMARY['identity_presence_by_class']['0']['present']:,} of {SUMMARY['target_counts']['0']:,} legitimate rows ({SUMMARY['identity_presence_by_class']['0']['present']/SUMMARY['target_counts']['0']:.2%}). This is an observed difference in coverage, not evidence that identity availability causes fraud.",
      f"Among card4 categories with at least 1,000 rows, discover has a {cards4['discover']['fraud_rate']:.2%} observed fraud rate (n={cards4['discover']['total']:,}); visa is {cards4['visa']['fraud_rate']:.2%} (n={cards4['visa']['total']:,}). The different group sizes affect how precisely the rates are estimated.",
      f"Among card6 categories with at least 1,000 rows, credit has a {cards6['credit']['fraud_rate']:.2%} observed fraud rate (n={cards6['credit']['total']:,}), compared with {cards6['debit']['fraud_rate']:.2%} for debit (n={cards6['debit']['total']:,}). This comparison is descriptive and unadjusted.",
    ]
    for i,text in enumerate(insight_texts,1):
        p=doc.add_paragraph(style='List Number');p.paragraph_format.space_after=Pt(6);p.paragraph_format.line_spacing=1.05;p.add_run(text)

    add_figure_page(doc,1,'Fraud and legitimate transaction counts','01_class_distribution.png',
      'This chart compares the number and share of legitimate and fraudulent transactions in the training data.',
      'Fraud represents about 3.5% of rows. The imbalance matters because a model could appear strong by predicting the majority class; later evaluation should report metrics that reflect fraud detection.',
      'Bar chart: 569,877 legitimate transactions (96.50%) and 20,663 fraudulent transactions (3.50%).')
    add_figure_page(doc,2,'Transaction amount distributions','02_amount_distribution.png',
      'The chart compares amount distributions by target on a log10 scale. Dashed vertical lines mark the median amount for each class.',
      'The median is slightly higher for the fraud group, while the distributions overlap substantially. The long tail makes the log scale useful for seeing both common amounts and less common high amounts.',
      'Overlaid density histograms of log10 transaction amounts for legitimate and fraudulent transactions; dashed lines mark class medians.')
    add_figure_page(doc,3,'Missing values across the most incomplete fields','03_missingness_top20.png',
      'This chart ranks the 20 fields with the largest missing percentages in the merged training table. Colors distinguish identity fields from transaction fields.',
      'Identity features dominate the top of the chart. Many transactions have no linked identity record, so these missing values can represent unavailable identity information rather than a lost transaction.',
      'Horizontal bars show missing percentages for the 20 most incomplete columns, grouped as identity or transaction features.')
    add_figure_page(doc,4,'Fraud rates by ProductCD','04_productcd_fraud_rate.png',
      'Each bar shows the fraction of transactions labeled fraud within a ProductCD category. Labels show category row counts; whiskers show Wilson 95% intervals.',
      'ProductCD C has the highest observed rate and W has the largest volume. The rates are useful for describing the data but are not causal effects or adjusted comparisons.',
      'Bar chart of observed fraud rate by ProductCD W, R, H, S, and C, with Wilson 95% intervals and sample counts.')
    add_figure_page(doc,5,'Fraud rates by card category','05_card_category_rates.png',
      'The panels compare observed rates by card brand (card4) and card type (card6). The chart includes categories with at least 1,000 transactions and shows Wilson 95% intervals.',
      'Discover and credit have higher observed rates than the most common categories in this sample. The chart is descriptive and does not account for differences in other transaction features.',
      'Two horizontal bar charts of fraud rates by card brand and card type, with Wilson 95% intervals and sample sizes.')

    doc.add_page_break();doc.add_heading('Conclusion',level=1)
    para(doc,'The Week 1 EDA confirms that the training target is highly imbalanced, transaction amounts have a long right tail, identity features are often missing, and observed fraud rates differ across ProductCD and card categories. These patterns can guide later analysis, but they do not establish causes or predictive value on their own.')
    para(doc,'Missing values and high-missingness fields were preserved for this descriptive work. No test data was used, no machine learning model was trained, and no Week 2 work was started.')
    doc.add_heading('Reproducible Files and Exact Locations',level=1)
    locs=[('EDA report',DOCX),('EDA script',ROOT/'eda_analysis.py'),('Word report builder',ROOT/'build_report.py'),('Findings summary',ROOT/'findings.md'),('Full EDA metrics',ROOT/'eda_summary.json'),('Per-column missingness',ROOT/'missingness_by_column.csv'),('ProductCD rates',ROOT/'productcd_fraud_rates.csv'),('card4 rates',ROOT/'card4_fraud_rates.csv'),('card6 rates',ROOT/'card6_fraud_rates.csv'),('Email-domain rates',ROOT/'email_domain_fraud_rates.csv'),('Class distribution chart',ROOT/'charts'/'01_class_distribution.png'),('Amount distribution chart',ROOT/'charts'/'02_amount_distribution.png'),('Missingness chart',ROOT/'charts'/'03_missingness_top20.png'),('ProductCD fraud-rate chart',ROOT/'charts'/'04_productcd_fraud_rate.png'),('Card category chart',ROOT/'charts'/'05_card_category_rates.png')]
    for label,path in locs:bullet(doc,f'{label}: {path.resolve()}')
    doc.add_heading('Reproduction Command',level=2)
    p=doc.add_paragraph();p.paragraph_format.left_indent=Inches(.2);r=p.add_run(f'python "{(ROOT/"eda_analysis.py").resolve()}"');r.font.name='Consolas';r.font.size=Pt(9);r.font.color.rgb=RGBColor.from_string(TEAL)
    para(doc,'The analysis script reads only the cleaned training table and writes charts and summaries under the week1_eda folder.')
    doc.save(DOCX)
    print(DOCX)

if __name__=='__main__':main()
