import asyncio
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

from nio import AsyncClient, MatrixRoom, RoomMessageText, LoginResponse, JoinResponse
from nio.exceptions import LocalProtocolError

from utils.logging_config import LoggerMixin
from utils.pydantic_models import TradingMode, HumanProposal


@dataclass
class Command:
    """Represents a parsed Matrix command."""
    name: str
    args: List[str]
    raw_message: str
    sender: str
    room_id: str
    timestamp: datetime


class MatrixClient(LoggerMixin):
    """
    Matrix client for bot communication with command processing.
    Handles all Matrix chat interactions and command parsing.
    """
    
    def __init__(
        self,
        homeserver: str,
        access_token: str,
        room_id: str,
        admin_users: List[str] = None,
        command_handlers: Dict[str, Callable] = None
    ):
        super().__init__()
        
        self.homeserver = homeserver
        self.access_token = access_token
        self.room_id = room_id
        self.admin_users = admin_users or []
        self.command_handlers = command_handlers or {}
        
        # Matrix client
        self.client = AsyncClient(homeserver, "")
        
        # State tracking
        self.is_connected = False
        self.current_mode = TradingMode.CONSERVATIVE
        self.auto_approve_enabled = False
        self.auto_approve_conditions = {}
        self.brief_schedule = "07:00"
        self.brief_timezone = "Europe/Ljubljana"
        self.brief_symbols = ["BTCUSDT", "ETHUSDT"]
        
        # Command patterns
        self.command_patterns = {
            'help': r'^/help$',
            'mode': r'^/mode\s+(aggressive|conservative|paper|halt)$',
            'profile_load': r'^/profile\s+load\s+(\w+)$',
            'set_risk': r'^/set\s+risk_per_trade\s+([\d.]+)$',
            'set_max_loss': r'^/set\s+max_daily_loss\s+([\d.]+)$',
            'set_heat_cap': r'^/set\s+heat_cap\s+([\d.]+)$',
            'set_leverage': r'^/set\s+leverage_cap\s+(\d+)$',
            'auto_on': r'^/auto\s+on\s+size<=(\d+)\s+confidence>=([\d.]+)$',
            'auto_off': r'^/auto\s+off$',
            'approve': r'^/approve\s+([a-zA-Z0-9-]+)$',
            'reject': r'^/reject\s+([a-zA-Z0-9-]+)$',
            'exit': r'^/exit\s+([A-Z]+)$',
            'move_sl': r'^/move_sl\s+([A-Z]+)\s+([\d.]+)$',
            'flatten': r'^/flatten$',
            'halt': r'^/halt$',
            'resume': r'^/resume$',
            'portfolio': r'^/portfolio$',
            'recent': r'^/recent$',
            'brief': r'^/brief$',
            'brief_schedule': r'^/brief\s+schedule\s+(\d{2}:\d{2})$',
            'brief_symbols_add': r'^/brief\s+symbols\s+add\s+([A-Z]+)$',
            'brief_symbols_remove': r'^/brief\s+symbols\s+remove\s+([A-Z]+)$',
            'brief_symbols_list': r'^/brief\s+symbols\s+list$',
        }
        
        self.logger.info("Matrix client initialized", 
                        homeserver=homeserver,
                        room_id=room_id,
                        admin_users=len(admin_users))
    
    async def connect(self):
        """Connect to Matrix server."""
        try:
            self.client.access_token = self.access_token
            
            # Verify connection by syncing
            sync_response = await self.client.sync(timeout=10000)
            
            if not sync_response.transport_response.ok:
                raise Exception(f"Failed to sync: {sync_response.transport_response}")
            
            self.is_connected = True
            
            # Set up event callbacks
            self.client.add_event_callback(self._handle_message, RoomMessageText)
            
            # Join the specified room if not already joined
            await self._ensure_room_joined()
            
            # Start sync loop
            asyncio.create_task(self._sync_loop())
            
            self.logger.info("Matrix client connected and synced")
            await self.send_message("🤖 Trading bot connected and ready!")
            
        except Exception as e:
            self.logger.error("Failed to connect to Matrix", error=str(e))
            self.is_connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from Matrix server."""
        if self.client:
            await self.client.close()
        self.is_connected = False
        self.logger.info("Matrix client disconnected")
    
    async def _ensure_room_joined(self):
        """Ensure bot has joined the specified room."""
        try:
            join_response = await self.client.join(self.room_id)
            if isinstance(join_response, JoinResponse):
                self.logger.info("Joined Matrix room", room_id=self.room_id)
            else:
                self.logger.warning("Failed to join room", room_id=self.room_id, response=join_response)
        except Exception as e:
            self.logger.error("Error joining room", room_id=self.room_id, error=str(e))
    
    async def _sync_loop(self):
        """Main sync loop for Matrix events."""
        try:
            while self.is_connected:
                sync_response = await self.client.sync(timeout=30000)
                if not sync_response.transport_response.ok:
                    self.logger.error("Sync failed", status=sync_response.transport_response.status_code)
                    break
        except Exception as e:
            self.logger.error("Sync loop error", error=str(e))
            self.is_connected = False
    
    async def _handle_message(self, room: MatrixRoom, event: RoomMessageText):
        """Handle incoming Matrix messages."""
        # Ignore our own messages
        if event.sender == self.client.user_id:
            return
        
        # Only process messages from the configured room
        if room.room_id != self.room_id:
            return
        
        message_body = event.body.strip()
        
        # Only process commands (messages starting with /)
        if not message_body.startswith('/'):
            return
        
        self.logger.info("Received command", 
                        sender=event.sender,
                        message=message_body,
                        room_id=room.room_id)
        
        # Parse and execute command
        try:
            command = await self._parse_command(message_body, event.sender, room.room_id)
            if command:
                await self._execute_command(command)
        except Exception as e:
            self.logger.error("Error processing command", 
                            message=message_body,
                            error=str(e))
            await self.send_message(f"❌ Error processing command: {str(e)}")
    
    async def _parse_command(self, message: str, sender: str, room_id: str) -> Optional[Command]:
        """Parse a command message."""
        for cmd_name, pattern in self.command_patterns.items():
            match = re.match(pattern, message, re.IGNORECASE)
            if match:
                return Command(
                    name=cmd_name,
                    args=list(match.groups()) if match.groups() else [],
                    raw_message=message,
                    sender=sender,
                    room_id=room_id,
                    timestamp=datetime.now(timezone.utc)
                )
        
        # No pattern matched
        await self.send_message(f"❓ Unknown command: `{message}`. Type `/help` for available commands.")
        return None
    
    async def _execute_command(self, command: Command):
        """Execute a parsed command."""
        # Check if user is authorized (for admin commands)
        admin_commands = {'mode', 'profile_load', 'set_risk', 'set_max_loss', 'set_heat_cap', 
                         'set_leverage', 'auto_on', 'auto_off', 'flatten', 'halt', 'resume', 
                         'brief_schedule', 'brief_symbols_add', 'brief_symbols_remove'}
        
        if command.name in admin_commands and command.sender not in self.admin_users:
            await self.send_message(f"❌ Unauthorized. Admin access required for `{command.raw_message}`")
            return
        
        # Execute command
        if command.name in self.command_handlers:
            try:
                await self.command_handlers[command.name](command)
            except Exception as e:
                self.logger.error("Command handler error", command=command.name, error=str(e))
                await self.send_message(f"❌ Error executing `{command.raw_message}`: {str(e)}")
        else:
            # Built-in command handling
            await self._handle_builtin_command(command)
    
    async def _handle_builtin_command(self, command: Command):
        """Handle built-in commands."""
        if command.name == 'help':
            await self._handle_help_command()
        
        elif command.name == 'mode':
            mode = command.args[0].upper()
            self.current_mode = TradingMode(mode)
            await self.send_message(f"🔄 Trading mode set to: **{mode}**")
        
        elif command.name == 'auto_off':
            self.auto_approve_enabled = False
            self.auto_approve_conditions = {}
            await self.send_message("⏸️ Auto-approval **disabled**")
        
        elif command.name == 'auto_on':
            size_limit = float(command.args[0])
            confidence_threshold = float(command.args[1])
            self.auto_approve_enabled = True
            self.auto_approve_conditions = {
                'max_size_usd': size_limit,
                'min_confidence': confidence_threshold
            }
            await self.send_message(f"✅ Auto-approval **enabled** - Size ≤ ${size_limit}, Confidence ≥ {confidence_threshold}")
        
        elif command.name == 'brief_schedule':
            self.brief_schedule = command.args[0]
            await self.send_message(f"📅 Brief schedule set to: **{self.brief_schedule}** ({self.brief_timezone})")
        
        elif command.name == 'brief_symbols_add':
            symbol = command.args[0]
            if symbol not in self.brief_symbols:
                self.brief_symbols.append(symbol)
                await self.send_message(f"➕ Added **{symbol}** to brief coverage")
            else:
                await self.send_message(f"ℹ️ **{symbol}** already in brief coverage")
        
        elif command.name == 'brief_symbols_remove':
            symbol = command.args[0]
            if symbol in self.brief_symbols:
                self.brief_symbols.remove(symbol)
                await self.send_message(f"➖ Removed **{symbol}** from brief coverage")
            else:
                await self.send_message(f"ℹ️ **{symbol}** not in brief coverage")
        
        elif command.name == 'brief_symbols_list':
            symbols_list = ", ".join(self.brief_symbols)
            await self.send_message(f"📋 **Brief Symbols**: {symbols_list}")
        
        elif command.name == 'halt':
            self.current_mode = TradingMode.HALT
            await self.send_message("🛑 **HALT MODE ACTIVATED** - All trading stopped")
        
        elif command.name == 'resume':
            self.current_mode = TradingMode.CONSERVATIVE
            await self.send_message("▶️ **RESUMED** - Trading mode: Conservative")
        
        else:
            await self.send_message(f"⚠️ Command `{command.name}` recognized but not implemented yet")
    
    async def _handle_help_command(self):
        """Handle /help command."""
        help_text = """
🤖 **Trading Bot Commands**

**Mode & Profile:**
• `/mode <aggressive|conservative|paper|halt>` - Set trading mode
• `/profile load <name>` - Load risk profile

**Risk Settings:**
• `/set risk_per_trade <pct>` - Risk per trade (e.g., 0.02 = 2%)
• `/set max_daily_loss <pct>` - Max daily loss limit
• `/set heat_cap <pct>` - Portfolio heat cap
• `/set leverage_cap <x>` - Max leverage multiplier

**Auto-Approval:**
• `/auto on size<=<usd> confidence>=<x>` - Enable auto-approval
• `/auto off` - Disable auto-approval

**Trade Management:**
• `/approve <decision_id>` - Approve a trade proposal
• `/reject <decision_id>` - Reject a trade proposal
• `/exit <SYMBOL>` - Exit position immediately
• `/move_sl <SYMBOL> <price>` - Move stop loss

**Control:**
• `/flatten` - Close all positions
• `/halt` - Stop all trading
• `/resume` - Resume trading

**Information:**
• `/portfolio` - Show portfolio status
• `/recent` - Show recent trades

**Daily Brief:**
• `/brief` - Generate immediate brief
• `/brief schedule <HH:MM>` - Set brief time
• `/brief symbols add|remove <SYMBOL>` - Manage coverage
• `/brief symbols list` - Show covered symbols

**Current Status:**
• **Mode:** {mode}
• **Auto-approval:** {auto_status}
• **Brief schedule:** {brief_time} ({tz})
• **Brief symbols:** {symbols}
        """.format(
            mode=self.current_mode.value.title(),
            auto_status="Enabled" if self.auto_approve_enabled else "Disabled",
            brief_time=self.brief_schedule,
            tz=self.brief_timezone,
            symbols=", ".join(self.brief_symbols)
        )
        
        await self.send_message(help_text)
    
    async def send_message(self, message: str, formatted: bool = True):
        """Send a message to the configured room."""
        try:
            if formatted:
                await self.client.room_send(
                    room_id=self.room_id,
                    message_type="m.room.message",
                    content={
                        "msgtype": "m.text",
                        "body": message,
                        "format": "org.matrix.custom.html",
                        "formatted_body": message.replace("**", "<strong>").replace("**", "</strong>")
                                                .replace("*", "<em>").replace("*", "</em>")
                                                .replace("`", "<code>").replace("`", "</code>")
                                                .replace("\n", "<br/>")
                    }
                )
            else:
                await self.client.room_send(
                    room_id=self.room_id,
                    message_type="m.room.message",
                    content={
                        "msgtype": "m.text",
                        "body": message
                    }
                )
            
            self.logger.debug("Message sent to Matrix", length=len(message))
            
        except Exception as e:
            self.logger.error("Failed to send Matrix message", error=str(e))
    
    async def send_proposal(self, proposal: HumanProposal):
        """Send a trade proposal to Matrix."""
        try:
            # Format the proposal message
            message = f"""
🎯 **TRADE PROPOSAL** #{proposal.decision_id[:8]}

**{proposal.symbol}** - {proposal.decision.action.value}
**Confidence:** {proposal.decision.confidence:.1%}
**Position Size:** {proposal.decision.position_size_pct:.1%} of portfolio

{proposal.summary_text}

**Entry:** {proposal.decision.entry_type.value}
{f"**Price:** ${proposal.decision.entry_price:.6f}" if proposal.decision.entry_price else "**Price:** Market"}
{f"**Stop Loss:** ${proposal.decision.stop_loss_price:.6f}" if proposal.decision.stop_loss_price else ""}
{f"**Take Profit:** ${proposal.decision.take_profit_price:.6f}" if proposal.decision.take_profit_price else ""}

**Reasoning:** {proposal.decision.reasoning}

{'🤖 **Auto-approved**' if not proposal.requires_approval else '⏳ **Awaiting approval** - Reply with `/approve ' + proposal.decision_id[:8] + '` or `/reject ' + proposal.decision_id[:8] + '`'}
            """
            
            await self.send_message(message)
            
        except Exception as e:
            self.logger.error("Failed to send proposal", proposal_id=proposal.decision_id, error=str(e))
    
    async def send_brief(self, brief_content: str):
        """Send a daily brief to Matrix."""
        try:
            await self.send_message(brief_content)
        except Exception as e:
            self.logger.error("Failed to send brief", error=str(e))
    
    async def send_alert(self, alert_type: str, message: str, urgent: bool = False):
        """Send an alert message."""
        emoji = "🚨" if urgent else "⚠️"
        alert_message = f"{emoji} **{alert_type.upper()}**\n{message}"
        await self.send_message(alert_message)
    
    def register_command_handler(self, command: str, handler: Callable):
        """Register a custom command handler."""
        self.command_handlers[command] = handler
        self.logger.info("Command handler registered", command=command)
    
    def should_auto_approve(self, proposal: HumanProposal, position_size_usd: float) -> bool:
        """Check if a proposal should be auto-approved."""
        if not self.auto_approve_enabled:
            return False
        
        conditions = self.auto_approve_conditions
        
        # Check size limit
        if position_size_usd > conditions.get('max_size_usd', 0):
            return False
        
        # Check confidence threshold
        if proposal.decision.confidence < conditions.get('min_confidence', 1.0):
            return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current Matrix client status."""
        return {
            'connected': self.is_connected,
            'room_id': self.room_id,
            'current_mode': self.current_mode.value,
            'auto_approve_enabled': self.auto_approve_enabled,
            'auto_approve_conditions': self.auto_approve_conditions,
            'brief_schedule': self.brief_schedule,
            'brief_symbols': self.brief_symbols,
            'admin_users_count': len(self.admin_users)
        }