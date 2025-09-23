"""Main entry point for web automation agent."""
import asyncio
import os
from typing import Dict, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_core import WebAgent
from tool_manager import ToolManager
from utils import analyze_goal

import os
os.environ["GOOGLE_API_KEY"] = ""  # Replace with your actual API key
#
google_key = os.getenv("GOOGLE_API_KEY")


async def run_automation(goal: str, config: Optional[Dict[str, Any]] = None):
    """Run web automation with the given goal."""
    config = config or {}
    
    # Set up MCP client with headless mode and better browser management
    server_params = StdioServerParameters(
        command="/opt/homebrew/bin/npx",
        args=["-y", "@playwright/mcp@latest", "--isolated"],
    )
    
    print("🔌 Connecting to Playwright MCP server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ MCP session initialized")
            
            # Initialize tool manager
            tool_manager = ToolManager(session)
            await tool_manager.initialize_tools()
            print(f"🧰 Loaded {len(tool_manager.tools)} tools")
            
            # Get API key from environment or config
            api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Missing Google API key. Set GOOGLE_API_KEY environment variable.")
                
            # Initialize LLM
            model_name = config.get("model_name") or os.getenv("MODEL_NAME", "gemini-2.5-flash")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.2,
            ).bind_tools(tool_manager.tools)
            
            # Create and run agent
            agent = WebAgent(llm, tool_manager)
            
            # Analyze goal for better context
            goal_analysis = analyze_goal(goal)
            print(f"🎯 Goal analysis: {goal_analysis}")
            
            # Run the agent
            print(f"🤖 Starting automation for goal: {goal}")
    #        messages = await agent.run(goal)
           
            try:
                # Run the agent with just the goal parameter
                messages = await agent.run(goal)
            except asyncio.TimeoutError:
                print("❌ Agent run timed out!")
                return []
            except Exception as e:
                print(f"❌ Agent run failed: {str(e)}")
                return []
                
            # Print summary
            print("\n==== Automation Complete ====")
            last_msg = messages[-1] if messages else None
            if last_msg:
                print(f"Final status: {getattr(last_msg, 'content', 'No content')}")
            
            # Return final messages for analysis
            return messages
        

#Data to be filled
user_data = [
    {
      "name": "John Doe",
      "email": "john.doe6@example.com",
      "Date of Birth,":"01/08/1995, (DD/MM/YYYY format)",
      "Date of Joining,":"01/10/2025, (DD/MM/YYYY format)",
      "phone": "+91-9876543210",
      "Address fields":"123 Main St, City, State 12345",
      "City, Country":"Hyderabad, India",
      "website": "www.johndoe.dev",
      "linkedin": "https://linkedin.com/in/example",
      "github": "https://github.com/example",
      "Work link/Portfolio" : "https://example.com",
      "languages":["English","Hindi"],
      "education":
            {
                "university_name": "University of Borosil",
                "degree": "Master of Science",
                "major": "Computer Science",
                "start_date": "2018",
                "end_date": "2023",
            },
      "experiences":
            {
                "company_name": "ABC Corp",
                "title": "Software Engineer",
                "city": "Banglore",
                "state": "Karnataka",
                "start_date": "Jan 2018",
                "end_date": "Dec 2022",
            },
      "skills":["Python", "Java", "C++", "C#", "SQL", "NoSQL", "Data Structures & Algorithms", "RESTful API Development", "Git/GitHub", "Statistics & Probability", "Machine Learning", "Deep Learning"],
      "summary": "A highly skilled Java Developer with expertise in designing, developing, and maintaining Java-based applications."
    }
  ]

async def main():


    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Web Automation Agent")
    parser.add_argument("goal", nargs="*", help="User goal for web automation")
    parser.add_argument("--api-key", help="Google API key (overrides environment variable)")
    parser.add_argument("--model", help="Model name (default: gemini-2.5-flash)")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum execution steps")
    
    args = parser.parse_args()
   
    # Default goal if none provided
    goal = "".join(args.goal) if args.goal else (
        "Navigate to https://boards.greenhouse.io/embed/job_app?token=6114125&gh_src=be8ebc4b1&source=Linkedln , get all the job query form fields and fill them with appropriate data and submit the form. "
    )

    config = { 
        "api_key": args.api_key,
        "model_name": args.model,
        "max_steps": args.max_steps
        
    }
    
    await run_automation(goal, config)


if __name__ == "__main__":
    asyncio.run(main())

#SHIVnND FORMS
# https://www.michaelpage.com/job-apply-external/ref/jn-092025-6830537/ZXZhbGVldG9ycmVzLjMxMjk3LjE1NTBAcGFnZXVzYS5hcGxpdHJhay5jb20%3D?utm_source=LinkedIn&utm_medium=Job%20board&utm_campaign=LinkedIn%20job%20posting&land=external&_exception_statuscode=404&success=true&type=standard&job_alert=1&hash=a8f1e3362413747476ed2b8257c48abd&apply_questions=true&source=LinkedIn
### https://careers-marzetti.icims.com/jobs/7332/qa-analyst-i/job?mode=submit_apply
# https://www.michaelpage.com/job-apply-external/ref/jn-092025-6830537/ZXZhbGVldG9ycmVzLjMxMjk3LjE1NTBAcGFnZXVzYS5hcGxpdHJhay5jb20%3D?utm_source=LinkedIn&utm_medium=Job%20board&utm_campaign=LinkedIn%20job%20posting&land=external&_exception_statuscode=404&success=true&type=standard&job_alert=1&hash=5c9e4aa144c01ad3f955de142224a065&apply_questions=true&source=LinkedIn
# https://www.glassdoor.co.in/job-listing/quality-assurance-analyst-docmagic-JV_IC1146894_KO0,25_KE26,34.htm?jl=1009867597964&src=GD_JOB_AD&uido=4D50FDA29824EC79A40F8E33C8B78A4C&ao=1136043&jrtk=5-yul1-3-1j5dp09l2gef7801-1j59t1dgli95b800&cs=1_e5b47475&s=335&t=JA&pos=102&guid=0000019953d0b57d809cfe60a34c855f&jobListingId=1009867597964&ea=1&vt=e&cb=1758177799543&ctt=1758178091101
# https://www.glassdoor.co.in/job-listing/product-quality-analyst-open-healthcare-us-JV_IC1146894_KO0,23_KE24,42.htm?jl=1009822342093&src=GD_JOB_AD&uido=4D50FDA29824EC79A40F8E33C8B78A4C&ao=1110586&jrtk=5-yul1-3-1j5dnuqj0l029801-1j59t1dgli95b800---6NYlbfkN0AuEUzteMmnHzPWg-NNL8sVAgxxksUHooZcGoH29JCOXfYUCkHiTwf27mlx6Nax3s3Y43RvEcox_QzKCMLZVLdANorAWP3eSz7IPAGAzQyCZXoCMiygzULDvx00_3CgUq2hj07QURTwAwst8Sq2KvFuR3GaUmnoqR6MlzkFbTtYwjuS68b5lNHrSO_pW3ocS8gCPH2YmWVB3yCN9ojBzoqJnmi1wFplwF-s3yo1m69tfXjuEt38C4VevENXiTkiq7zR_0bvLqX99OCzkSFn7_VRo8fCvMM56qGhbBU4-FvA36zBTTWq4X25vs6rZdU_KVn6bf1h_pQJhXInN1tWJxhhWQfeQ2JQ4qpw9R7tS0t9qnh_9Vl6ieiMpojOSK1BLGqTVesObyxyQFJ0Sxq3IM272geOQV97mNH4v3JRLlvt5JyxXmjlGv56KRn6nTrk8JFCKrffStq3Lv2kxZwwEIP_yegqHI-l2QqWpDJ7Yobg3VrRsphS1BYL59LJoGix7invhOicnOJxN01J4l3dvNkIvYPa_sS25J1l3VpyLBJ2SslC6NkNmdQMTZCkNxsI3pIsOsHNK7zVU24m1H6aOV4Q90V4CYNGZaTRMA9tLdfvZjGQPAmBOEyMOooggfBc-C6xwc31gXvGpZg7Er7jVnFXLH3ZBZn2lx6QMcCoQlOEIFwqPU6ZS7bfFttxlkpgidKpFv1xuvp3lqxcUmuyJ1HN0vm5PNq90FsFjfKgYM1_G2yIopeevJxp&cs=1_bb2c8fab&s=335&t=JA&pos=101&cpc=217C45A42544DB93&guid=0000019953d0b57d809cfe60a34c855f&jobListingId=1009822342093&ea=1&vt=e&cb=1758176702112&ctt=1758176828648
# https://www.michaelpage.com/job-apply-external/ref/jn-092025-6830537/ZXZhbGVldG9ycmVzLjMxMjk3LjE1NTBAcGFnZXVzYS5hcGxpdHJhay5jb20%3D?utm_source=LinkedIn&utm_medium=Job%20board&utm_campaign=LinkedIn%20job%20posting&land=external&_exception_statuscode=404&uact=job-apply&success=true&type=standard&job_alert=1&hash=690df9b900c8f724645f00869da636cd&apply_questions=true&source=LinkedIn
# https://phf.tbe.taleo.net/phf04/ats/careers/v2/thankYou?org=FOREMOST&cws=37&rid=3481
# https://us.smartapply.indeed.com/beta/indeedapply/form/post-apply




## https://careers.blueyonder.com/us/en/apply?jobSeqNo=BYPBYXUS253418EXTERNALENUS&utm_source=linkedin&utm_medium=phenom-feeds&step=1&stepname=personalInformation


#https://careers.intimetec.in/intimetec/jobview/agentic-ai-engineer-bangalore-karnataka-india-jaipur-rajasthan-india-lang-chain-crew-ai-lang-graph-google-adk-llms-openai-llama-claude-gemini-etc-2025072214584469?source=linkedin
#file:///Users/dineshk/Downloads/simplified_job_application.html
# https://form.jotform.com/252433290053449
#https://form.jotform.com/252131352654450
#https://job.10xscale.ai/4846461985313787904
#file:///Users/dineshk/Downloads/text-only-job-application.html
# https://form.jotform.com/252432911548457
# https://form.jotform.com/252433196396464
# https://form.jotform.com/252441958067464


# apply button
# https://aurigait.keka.com/careers/jobdetails/57153
# https://careers.intimetec.in/intimetec/jobview/agentic-ai-engineer-bangalore-karnataka-india-jaipur-rajasthan-india-lang-chain-crew-ai-lang-gr
# https://www.rubrik.com/company/careers/departments/job.6502559?gh_jid=6502559&gh_src=2e352a8a1us


# NEXT
# https://secure6.saashr.com/ta/6183000.careers?ShowJob=637934976&source=LinkedIn


#1 https://boards.greenhouse.io/embed/job_app?token=6114125&gh_src=be8ebc4b1&source=Linkedln
#2 https://docs.google.com/forms/d/e/1FAIpQLSe8ANnmECPWFLkmRwmF6QCVXQLwGR-PrvU99JQiLhLc9Ye-oQ/viewform?usp=header
#3 https://aurigait.keka.com/careers/jobdetails/57153
#4 https://www.globallogic.com/careers/senior-data-scientist-irc273505/?utm_source=LinkedIn&utm_medium=jobboard&utm_campaign=career#apply_now
#5 https://careers.intimetec.in/intimetec/jobview/agentic-ai-engineer-bangalore-karnataka-india-jaipur-rajasthan-india-lang-chain-crew-ai-lang-graph-google-adk-llms-openai-llama-claude-gemini-etc-2025072214584469?source=linkedin


# JOT FORM
# https://form.jotform.com/252442272934457

#       https://ibqbjb.fa.ocs.oraclecloud.com/hcmUI/CandidateExperience/en/sites/Honeywell/job/115902?utm_medium=jobboard&utm_source=linkedin

