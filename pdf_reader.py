from PyPDF2 import PdfReader

INPUT = '''We see ChatGPT as an engine that will eventually power human interactions with computer systems in a 
familiar, natural, and intuitive way. As ChatGPT stated, large language models can be put to work as a communication 
engine in a variety of applications across a number of vertical markets.
Glaringly absent in its answer is the use of ChatGPT in search engines. Microsoft, which is an investor in OpenAI, is
integrating ChatGPT into its Bing search engine. The use of a large language model enables more complex and more
natural searches and extract deeper meaning and better context from source material. This is ultimately expected 
to deliver more robust and useful results.
Is AI coming for your job?
Every wave of new and disruptive technology has incited fears of mass job losses due to automation, and we are already
seeing those fears expressed relative to AI generally and ChatGPT specifically. The year 1896, when Henry Ford rolled out
his first automobile, was probably not a good year for buggy whip makers. When IBM introduced its first mainframe, the
System/360, in 1964, office workers feared replacement by mechanical brains that never made mistakes, never called in
sick, and never took vacations.
There are certainly historical cases of job displacement due to new technology adoption, and ChatGPT may unseat some
office workers or customer service reps. However, we think AI tools broadly will end up as part of the solution in an
economy that has more job openings than available workers.
However, economic history shows that technology of any sort (i.e., manufacturing technology, communications
technology, information technology) ultimately makes productive workers more productive and is net additive to
employment and economic growth.
How big is the opportunity?
The broad AI hardware and services market was nearly USD 36bn in 2020, based on IDC and Bloomberg Intelligence data.
We expect the market to grow by 20% CAGR to reach USD 90bn by 2025. Given the relatively early monetization stage
of conversational AI, we estimate that the segment accounted for 10% of the broader AI’s addressable market in 2020,
predominantly from enterprise and consumer subscriptions.
That said, user adoption is rapidly rising. ChatGPT reached its first 1 million user milestone in a week, surpassing Instagram
to become the quickest application to do so. Similarly, we see strong interest from enterprises to integrate conservational
AI into their existing ecosystem. As a result, we believe conversational AI’s share in the broader AI’s addressable market
can climb to 20% by 2025 (USD 18–20bn). Our estimate may prove to be conservative; they could be even higher
if conversational AI improvements (in terms of computing power, machine learning, and deep learning capabilities),
availability of talent, enterprise adoption, spending from governments, and incentives are stronger than expected.
How to invest in AI?
We see artificial intelligence as a horizontal technology that will have important use cases across a number of applications
and industries. From a broader perspective, AI, along with big data and cybersecurity, forms what we call the ABCs of
technology. We believe these three major foundational technologies are at inflection points and should see faster adoption
over the next few years as enterprises and governments increase their focus and investments in these areas.
Conservational AI is currently in its early stages of monetization and costs remain high as it is expensive to run. Instead
of investing directly in such platforms, interested investors in the short term can consider semiconductor companies, and
cloud-service providers that provides the infrastructure needed for generative AI to take off. In the medium to long term,
companies can integrate generative AI to improve margins across industries and sectors, such as within healthcare and
traditional manufacturing.
Outside of public equities, investors can also consider opportunities in private equity (PE). We believe the tech sector is
currently undergoing a new innovation cycle after 12–18 months of muted activity, which provides interesting and new
opportunities that PE can capture through early-stage investments.
'''

# This function is reading PDF from the start page to final page
# given as input (if less pages exist, then it reads till this last page)
def get_pdf_text(document_path, start_page=1, final_page=999):
    return INPUT
    # reader = PdfReader(document_path)
    # number_of_pages = len(reader.pages)
    # print(f'numer of page: {number_of_pages}')
    # page = '\npage#' + str(start_page) + '\n'
    # for page_num in range(start_page - 1, min(number_of_pages, final_page)):
    #     page += reader.pages[page_num].extract_text()
    #     page += '\n\npage#' + str(page_num + 2) + ' \n' if page_num < final_page - 1 else ''
    # return page


if __name__ == '__main__':
    doc_path_name = 'documents/chat_gpt_ubs.pdf'
    doc_text = get_pdf_text(doc_path_name, 1, 2)
    print(doc_text)
