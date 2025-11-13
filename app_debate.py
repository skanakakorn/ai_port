import os
import json
import time
import re
import streamlit as st
import requests

# ============== LLM Adapter Function ===================
def call_llm_debate(messages, provider=None, model=None, temperature=0.7, max_tokens=300):
    """
    Call LLM with messages list. Supports GROQ and OPENAI.
    messages: list of dicts with "role" and "content" keys
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "GROQ").upper()
    if model is None:
        model = os.getenv("LLM_MODEL_ID", "openai/gpt-oss-120b")
    
    if provider == "OPENAI":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Newer OpenAI models use max_completion_tokens instead of max_tokens
            # Some newer models only support default temperature (1.0)
            params = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": max_tokens,
            }
            # Try with temperature first (if not default)
            if temperature != 1.0:
                params["temperature"] = temperature
            
            try:
                resp = client.chat.completions.create(**params)
                content = resp.choices[0].message.content.strip()
                # Check if response was cut off (finish_reason == "length")
                if resp.choices[0].finish_reason == "length":
                    # Response was cut off due to token limit
                    content += "\n\n[หมายเหตุ: ข้อความถูกตัดเนื่องจากถึงขีดจำกัดความยาว]"
                return content
            except Exception as temp_error:
                # If temperature is not supported, retry without it
                error_str = str(temp_error)
                if "temperature" in error_str.lower() and "unsupported" in error_str.lower():
                    params.pop("temperature", None)  # Remove temperature if present
                    resp = client.chat.completions.create(**params)
                    content = resp.choices[0].message.content.strip()
                    if resp.choices[0].finish_reason == "length":
                        content += "\n\n[หมายเหตุ: ข้อความถูกตัดเนื่องจากถึงขีดจำกัดความยาว]"
                    return content
                else:
                    raise temp_error
        except Exception as e:
            return f"[LLM error OPENAI: {e}]"
    
    elif provider == "GROQ":
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"].strip()
            # Check if response was cut off (finish_reason == "length")
            if data["choices"][0].get("finish_reason") == "length":
                # Response was cut off due to token limit
                content += "\n\n[หมายเหตุ: ข้อความถูกตัดเนื่องจากถึงขีดจำกัดความยาว]"
            return content
        except Exception as e:
            return f"[LLM error GROQ: {e}]"
    
    else:
        return "[DUMMY LLM] This is a placeholder response. Set LLM_PROVIDER and API key."

# ============== Agent System Prompts ===================
VALUE_INVESTOR_SYSTEM = """คุณเป็นนักลงทุนแบบ Value Investor ที่เชี่ยวชาญในการวิเคราะห์หุ้น
คุณเน้นที่:
- มูลค่าพื้นฐาน (fundamentals) เช่น P/E ratio, P/B ratio, book value
- เงินปันผล (dividends) และความมั่นคงทางการเงิน
- การประเมินราคาที่ต่ำกว่ามูลค่าจริง (undervalued)
- มุมมองระยะยาวและความเสี่ยงต่ำ
- งบการเงินที่แข็งแกร่งและหนี้สินต่ำ

ตอบเป็นภาษาไทยอย่างกระชับและชัดเจน มีวิธีการพูดแบบด็อกเตอร์นิเวศ เน้นเหตุผลเชิงมูลค่าและความมั่นคง
สำคัญ: 
- ตอบโดยตรงโดยไม่ต้องระบุชื่อตัวเองหรือใส่คำนำหน้าว่า "Value Investor" ในข้อความ
- อย่าซ้ำซ้อนกับสิ่งที่คุณเคยพูดไปแล้วในรอบก่อนหน้า
- ตอบโต้แย้งประเด็นที่ Growth Investor เพิ่งกล่าวมา และเสนอมุมมองใหม่"""

GROWTH_INVESTOR_SYSTEM = """คุณเป็นนักลงทุนแบบ Growth Investor ที่เชี่ยวชาญในการวิเคราะห์หุ้น
คุณเน้นที่:
- การเติบโตของรายได้ (revenue growth) และกำไร
- การขยายตัวของตลาดและนวัตกรรม
- ศักยภาพในอนาคตและแนวโน้มอุตสาหกรรม
- การลงทุนในบริษัทที่กำลังเติบโตอย่างรวดเร็ว
- มูลค่าตลาดที่เพิ่มขึ้น (market expansion)

ตอบเป็นภาษาไทยอย่างกระชับและชัดเจน เน้นโอกาสการเติบโตและศักยภาพ
สำคัญ: 
- ตอบโดยตรงโดยไม่ต้องระบุชื่อตัวเองหรือใส่คำนำหน้าว่า "Growth Investor" ในข้อความ
- อย่าซ้ำซ้อนกับสิ่งที่คุณเคยพูดไปแล้วในรอบก่อนหน้า
- ตอบโต้แย้งประเด็นที่ Value Investor เพิ่งกล่าวมา และเสนอมุมมองใหม่"""

# ============== Debate Engine ===================
class DebateEngine:
    def __init__(self, question, provider, model, max_duration=15):
        self.question = question
        self.provider = provider
        self.model = model
        self.max_duration = max_duration
        self.start_time = None
        self.messages = []
        self.debate_active = False
        self.debate_history = []
        
    def get_elapsed_time(self):
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
    
    def get_remaining_time(self):
        return max(0, self.max_duration - self.get_elapsed_time())
    
    def is_time_up(self):
        return self.get_elapsed_time() >= self.max_duration
    
    def add_message(self, role, content, agent_name):
        """Add message to debate history"""
        self.debate_history.append({
            "role": role,
            "content": content,
            "agent_name": agent_name,
            "timestamp": time.time() - self.start_time if self.start_time else 0
        })
    
    def get_agent_response(self, agent_name, agent_system, conversation_history):
        """Get response from an agent"""
        # Determine the other agent's name
        other_agent_name = "Growth Investor" if agent_name == "Value Investor" else "Value Investor"
        
        messages = [
            {"role": "system", "content": agent_system},
            {"role": "user", "content": f"คำถาม: {self.question}\n\nให้คุณตอบในฐานะ{agent_name} และโต้แย้งกับ{other_agent_name}"}
        ]
        
        # Extract what this agent has already said (to avoid repetition)
        my_previous_points = []
        other_agent_last_points = []
        
        if conversation_history:
            for msg in conversation_history:
                if msg["role"] == "assistant":
                    speaker = msg.get("agent_name", "Unknown")
                    if speaker == agent_name:
                        my_previous_points.append(msg['content'])
                    elif speaker == other_agent_name:
                        other_agent_last_points.append(msg['content'])
        
        # Add conversation history with agent names for context
        if conversation_history:
            history_text = "ประวัติการโต้วาที:\n"
            for msg in conversation_history:
                if msg["role"] == "assistant":
                    speaker = msg.get("agent_name", "Unknown")
                    history_text += f"[{speaker}]: {msg['content']}\n\n"
            messages.append({
                "role": "user",
                "content": history_text
            })
        
        # Build explicit instruction to avoid repetition
        instruction_parts = [
            f"ตอนนี้เป็นตาของคุณแล้ว ({agent_name}) ให้ตอบโต้แย้งกับ{other_agent_name}"
        ]
        
        # If this agent has spoken before, explicitly list what they've already said
        if my_previous_points:
            instruction_parts.append(f"\n⚠️ สิ่งที่คุณเคยพูดไปแล้ว (ห้ามซ้ำ!):")
            for i, point in enumerate(my_previous_points[-2:], 1):  # Show last 2 responses
                # Truncate if too long
                truncated = point[:200] + "..." if len(point) > 200 else point
                instruction_parts.append(f"  {i}. {truncated}")
        
        # If other agent just spoke, emphasize responding to their latest points
        if other_agent_last_points:
            latest_other = other_agent_last_points[-1]
            instruction_parts.append(f"\n🎯 {other_agent_name} เพิ่งกล่าวว่า:")
            instruction_parts.append(f"  \"{latest_other[:300]}{'...' if len(latest_other) > 300 else ''}\"")
            instruction_parts.append(f"\nคุณต้องตอบโต้แย้งประเด็นเหล่านี้โดยเฉพาะ!")
        
        instruction_parts.extend([
            "\nคำแนะนำสำคัญ:",
            "- อย่าซ้ำซ้อนกับสิ่งที่คุณเคยพูดไปแล้ว (ดูรายการด้านบน)",
            "- ตอบโต้แย้งประเด็นที่" + other_agent_name + "เพิ่งกล่าวมา",
            "- เสนอประเด็นใหม่หรือมุมมองที่แตกต่างจากที่เคยพูด",
            "- ตอบอย่างละเอียดและครบถ้วน (4-6 ประโยค)",
            "- ตอบโดยตรงโดยไม่ต้องระบุชื่อตัวเอง"
        ])
        
        messages.append({
            "role": "user",
            "content": "\n".join(instruction_parts)
        })
        
        response = call_llm_debate(
            messages,
            provider=self.provider,
            model=self.model,
            temperature=0.4,  # Increased from 0.2 to add more variation and reduce repetition
            max_tokens=1000  # Increased from 500 to prevent text cutoff
        )
        
        # Clean up response - remove any agent name prefixes that might have been added
        response = response.strip()
        # Remove patterns like "Value Investor:", "Growth Investor:", "Value Investor (ตอบโต้):", etc.
        # Also handle Thai variations
        patterns = [
            r'^(Value Investor|Growth Investor)\s*[:\-\(].*?\)?\s*',  # English with punctuation
            r'^(Value Investor|Growth Investor)\s+',  # English with space
            r'^.*?Value Investor.*?[:\-]\s*',  # Any text before "Value Investor:"
            r'^.*?Growth Investor.*?[:\-]\s*',  # Any text before "Growth Investor:"
        ]
        for pattern in patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)
        response = response.strip()
        
        return response
    
    def run_debate(self):
        """Run the debate for max_duration seconds, with max 3 turns per side"""
        self.start_time = time.time()
        self.debate_active = True
        
        # Track turns for each agent (max 3 each)
        value_turns = 0
        growth_turns = 0
        max_turns_per_side = 3
        
        # Initial messages from both agents (counts as turn 1 for each)
        value_msg = self.get_agent_response(
            "Value Investor",
            VALUE_INVESTOR_SYSTEM,
            []
        )
        self.add_message("assistant", value_msg, "Value Investor")
        value_turns += 1
        time.sleep(1)  # Pause 1 second after Value Investor's response
        
        growth_msg = self.get_agent_response(
            "Growth Investor",
            GROWTH_INVESTOR_SYSTEM,
            self.debate_history
        )
        self.add_message("assistant", growth_msg, "Growth Investor")
        growth_turns += 1
        time.sleep(1)  # Pause 1 second after Growth Investor's response
        
        # Alternate turns until time is up or either side reaches max turns
        turn = 0
        while not self.is_time_up() and self.debate_active:
            # Check if either side has reached max turns
            if value_turns >= max_turns_per_side and growth_turns >= max_turns_per_side:
                break
            
            remaining = self.get_remaining_time()
            if remaining < 3:  # Not enough time for another turn
                break
            
            # Alternate between agents, but only if they haven't reached max turns
            if turn % 2 == 0:
                # Value Investor's turn
                if value_turns < max_turns_per_side:
                    response = self.get_agent_response(
                        "Value Investor",
                        VALUE_INVESTOR_SYSTEM,
                        self.debate_history
                    )
                    self.add_message("assistant", response, "Value Investor")
                    value_turns += 1
                    time.sleep(1)  # Pause 1 second after Value Investor's response
                elif growth_turns < max_turns_per_side:
                    # Skip Value Investor if they've reached max, continue with Growth
                    turn += 1
                    continue
                else:
                    break
            else:
                # Growth Investor's turn
                if growth_turns < max_turns_per_side:
                    response = self.get_agent_response(
                        "Growth Investor",
                        GROWTH_INVESTOR_SYSTEM,
                        self.debate_history
                    )
                    self.add_message("assistant", response, "Growth Investor")
                    growth_turns += 1
                    time.sleep(1)  # Pause 1 second after Growth Investor's response
                elif value_turns < max_turns_per_side:
                    # Skip Growth Investor if they've reached max, continue with Value
                    turn += 1
                    continue
                else:
                    break
            
            turn += 1
        
        self.debate_active = False
        return self.debate_history
    
    def generate_summary(self):
        """Generate summary/conclusion of the debate"""
        # Build conversation summary
        conversation_text = f"คำถาม: {self.question}\n\n"
        for msg in self.debate_history:
            conversation_text += f"[{msg['agent_name']}]: {msg['content']}\n\n"
        
        summary_prompt = f"""สรุปการโต้วาทีระหว่าง Value Investor และ Growth Investor เกี่ยวกับคำถาม: {self.question}

การโต้วาที:
{conversation_text}

ให้สรุปประเด็นหลักที่ทั้งสองฝ่ายเสนออย่างครบถ้วนและสมบูรณ์ โดยให้ความยาวของทั้งสองส่วนใกล้เคียงกัน (จำนวนคำ/ประโยคใกล้เคียงกัน):

**1. มุมมองของ Value Investor**
อธิบายอย่างละเอียด ประมาณ 4-6 ประโยค (เน้นมูลค่าพื้นฐาน, ความมั่นคง, งบการเงิน) ต้องครบถ้วนและจบประโยคสุดท้ายให้สมบูรณ์

**2. มุมมองของ Growth Investor**
อธิบายอย่างละเอียด ประมาณ 4-6 ประโยค (เน้นการเติบโต, ศักยภาพ, นวัตกรรม) ต้องมีความยาวใกล้เคียงกับ Value Investor และจบประโยคสุดท้ายให้สมบูรณ์

**3. ข้อสรุปที่สมดุล**
วิเคราะห์เปรียบเทียบและให้คำแนะนำ (4-6 ประโยค) ต้องจบประโยคสุดท้ายให้สมบูรณ์

⚠️ สำคัญมาก: 
- ให้ทั้ง Value Investor และ Growth Investor มีจำนวนคำและประโยคใกล้เคียงกัน
- ต้องตอบครบทั้ง 3 ส่วน (Value Investor, Growth Investor, ข้อสรุป)
- ต้องจบประโยคสุดท้ายของแต่ละส่วนให้สมบูรณ์ อย่าตัดข้อความกลางคัน
- อย่าหยุดกลางประโยค ต้องเขียนให้เสร็จสมบูรณ์

ตอบเป็นภาษาไทย อย่าตัดข้อความกลางคัน"""
        
        messages = [
            {"role": "system", "content": "คุณเป็นผู้สรุปการโต้วาทีการลงทุน ให้สรุปอย่างเป็นกลาง ครบถ้วนทั้งสองฝ่าย โดยให้ความยาวของทั้งสองส่วน (Value Investor และ Growth Investor) ใกล้เคียงกันมาก สำคัญมาก: ต้องตอบครบทั้ง 3 ส่วน (Value Investor, Growth Investor, ข้อสรุป) และต้องจบประโยคสุดท้ายของแต่ละส่วนให้สมบูรณ์ อย่าตัดข้อความกลางคัน อย่าหยุดกลางประโยค ต้องเขียนให้เสร็จสมบูรณ์"},
            {"role": "user", "content": summary_prompt}
        ]
        
        summary = call_llm_debate(
            messages,
            provider=self.provider,
            model=self.model,
            temperature=0.5,
            max_tokens=4000
        )
        return summary

# ============== Streamlit UI ===================
st.set_page_config(
    page_title="AI Debate: Value vs Growth Investor",
    page_icon="💬",
    layout="wide"
)

st.title("💬 AI Debate: Value Investor vs Growth Investor")
st.caption("ให้ AI สองตัวโต้วาทีกันเกี่ยวกับการลงทุน - ระยะเวลา 15 วินาที")

# Sidebar Configuration
st.sidebar.header("⚙️ การตั้งค่า")

provider_options = ["GROQ", "OPENAI"]
default_provider = os.getenv("LLM_PROVIDER", "GROQ").upper()
if default_provider not in provider_options:
    default_provider = "GROQ"

selected_provider = st.sidebar.selectbox(
    "เลือก LLM Provider",
    options=provider_options,
    index=provider_options.index(default_provider) if default_provider in provider_options else 0
)

# Model selection based on provider
if selected_provider == "GROQ":
    default_model = os.getenv("LLM_MODEL_ID", "openai/gpt-oss-120b")
    model_options = ["openai/gpt-oss-120b", "llama-3.1-8b-instruct", "llama-3.1-70b-instruct"]
    if default_model not in model_options:
        model_options.insert(0, default_model)
    selected_model = st.sidebar.selectbox(
        "เลือก Model (Groq)",
        options=model_options,
        index=0
    )
else:  # OPENAI
    default_model = os.getenv("LLM_MODEL_ID", "gpt-4o-mini")
    model_options = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    if default_model not in model_options:
        model_options.insert(0, default_model)
    selected_model = st.sidebar.selectbox(
        "เลือก Model (OpenAI)",
        options=model_options,
        index=0
    )

# API Key status
api_key_name = f"{selected_provider}_API_KEY"
api_key = os.getenv(api_key_name)
if not api_key:
    st.sidebar.warning(f"⚠️ ตั้งค่า {api_key_name} ใน environment variables")
else:
    st.sidebar.success(f"✅ {selected_provider} API Key พร้อมใช้งาน")

st.sidebar.divider()
st.sidebar.markdown("**ตัวอย่างคำถาม:**")
st.sidebar.markdown("- AI ตัวไหนจะชนะในปี 2026: NVIDIA หรือ BYD?")
st.sidebar.markdown("- ควรลงทุนในหุ้นเทคโนโลยีหรือหุ้นพลังงาน?")
st.sidebar.markdown("- TSMC vs Intel: ใครจะดีกว่าในระยะยาว?")

# Main UI
st.divider()

# Initialize session state
if "debate_history" not in st.session_state:
    st.session_state.debate_history = []
if "debate_engine" not in st.session_state:
    st.session_state.debate_engine = None
if "debate_running" not in st.session_state:
    st.session_state.debate_running = False
if "debate_summary" not in st.session_state:
    st.session_state.debate_summary = None

# Question input
question = st.text_input(
    "💭 ใส่คำถามของคุณ:",
    placeholder="เช่น: AI ตัวไหนจะชนะในปี 2026: NVIDIA หรือ BYD?",
    key="question_input"
)

col1, col2 = st.columns([1, 4])
with col1:
    start_button = st.button("🚀 เริ่มการโต้วาที", type="primary", disabled=st.session_state.debate_running)
with col2:
    if st.session_state.debate_running:
        st.info("⏳ กำลังโต้วาที...")

# Chat display area
chat_container = st.container()

# Display chat messages
def display_chat_messages():
    """Display all chat messages in bubble format"""
    with chat_container:
        if st.session_state.debate_history:
            for msg in st.session_state.debate_history:
                agent_name = msg.get("agent_name", "Unknown")
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", 0)
                
                if agent_name == "Value Investor":
                    with st.chat_message("user", avatar="💰"):
                        st.markdown(f"**Value Investor:** {content}")
                        st.caption(f"เวลา: {timestamp:.1f}s")
                elif agent_name == "Growth Investor":
                    with st.chat_message("assistant", avatar="🚀"):
                        st.markdown(f"**Growth Investor:** {content}")
                        st.caption(f"เวลา: {timestamp:.1f}s")

# Timer display (will be shown during debate)

# Run debate when button is clicked
if start_button and question and not st.session_state.debate_running:
    if not api_key:
        st.error(f"⚠️ กรุณาตั้งค่า {api_key_name} ใน environment variables")
    else:
        st.session_state.debate_running = True
        st.session_state.debate_history = []
        st.session_state.debate_summary = None
        
        # Create debate engine
        engine = DebateEngine(question, selected_provider, selected_model, max_duration=15)
        st.session_state.debate_engine = engine
        
        # Run debate
        try:
            with st.spinner("⏳ กำลังโต้วาที... (15 วินาที)"):
                history = engine.run_debate()
                st.session_state.debate_history = history
            
            # Generate summary
            with st.spinner("📝 กำลังสร้างสรุป..."):
                summary = engine.generate_summary()
                st.session_state.debate_summary = summary
            
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            st.session_state.debate_running = False
        
        st.rerun()

# Display existing messages
display_chat_messages()

# Show summary if available
if st.session_state.debate_summary:
    st.divider()
    st.subheader("📝 สรุปการโต้วาที")
    st.info(st.session_state.debate_summary)

